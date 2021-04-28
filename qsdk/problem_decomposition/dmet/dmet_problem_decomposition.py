"DOC STRING"

from enum import Enum
from functools import reduce
from pyscf.cc import CCSD
import scipy
#from pyscf import scf
import numpy as np

from . import _helpers as helpers
from ..problem_decomposition import ProblemDecomposition
from ..electron_localization import iao_localization, meta_lowdin_localization
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.electronic_structure_solvers import FCISolver, CCSDSolver, VQESolver


class Localization(Enum):
    """ Enumeration of the electron localization supported by DMET."""
    meta_lowdin= 0
    iao = 1

class DMETProblemDecomposition(ProblemDecomposition):
    """Employ DMET as a problem decomposition technique.

    DMET single-shot algorithm is used for problem decomposition technique.
    By default, CCSD is used as the electronic structure solver, and
    Meta-Lowdin is used for the localization scheme.
    Users can define other electronic structure solver such as FCI or
    VQE as an impurity solver. IAO scheme can be used instead of the Meta-Lowdin
    localization scheme, but it cannot be used for minimal basis set.

    Attributes:
        molecule (pyscf.gto.mol): the molecular system
        mean-field (optional): mean-field of molecular system
        electron_localization (Localization): A type of localization scheme. Default is Meta-Lowdin.
        electronic_structure_solvers

        optimizer (function handle): a function defining the classical optimizer and its behavior
        initial_var_params (str or array-like) : initial value for the classical optimizer
        backend_options (dict) : parameters to build the Simulator class (see documentation of agnostic_simulator)
        up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down
            Default, False has alternating spin up/down ordering.
        verbose (bool) : Flag for DMET verbosity .

        electronic_structure_solver (subclass of ElectronicStructureSolver): A type of electronic structure solver. Default is CCSD.
        electron_localization_method (string): A type of localization scheme. Default is IAO.
    """

    def __init__(self, opt_dict):

        default_ccsd_options = dict()
        default_fci_options = dict()
        default_vqe_options = {"qubit_mapping": "jw",
                               "initial_var_params": "ones",
                               "verbose": False}

        default_options = {"molecule": None, "mean_field": None, 
                           "electron_localization": Localization.meta_lowdin,
                           "fragment_atoms": list(), 
                           "fragment_solvers": "ccsd",
                           "initial_chemical_potential": 0.0,
                           "solvers_options": list(),
                           "verbose": False}

        self.builtin_localization = set(Localization)

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                raise KeyError(f"Keyword :: {k}, not available in DMETProblemDecomposition.")

        # Raise error/warnings if input is not as expected
        if not self.molecule:
            raise ValueError(f"A molecule object must be provided when instantiating DMETProblemDecomposition.")

        # Check if the number of fragment sites is equal to the number of atoms in the molecule
        if self.molecule.natm != sum(self.fragment_atoms):
            raise RuntimeError("The number of fragment sites is not equal to the number of atoms in the molecule")

        # Check that the number of solvers matches the number of fragments.
        # If a single string is detected, it is converted to a list.
        # If a list is detected, it must have the same length than the fragment_atoms.
        if isinstance(self.fragment_solvers, str):
            self.fragment_solvers = [self.fragment_solvers for _ in range(len(self.fragment_atoms))]
        elif isinstance(self.fragment_solvers, list):
            if len(self.fragment_solvers) != len(self.fragment_atoms):
                raise RuntimeError("The number of solvers does not match the number of fragments.")

        # Check that the number of solvers options matches the number of solvers.
        if not self.solvers_options:
            for solver in self.fragment_solvers:
                if solver == "ccsd":
                    self.solvers_options.append(default_ccsd_options)
                elif solver == "fci":
                    self.solvers_options.append(default_fci_options)
                elif solver == "vqe":
                    self.solvers_options.append(default_vqe_options)
        elif isinstance(self.solvers_options, dict):
            self.solvers_options = [self.solvers_options for _ in self.fragment_solvers]
        elif isinstance(self.solvers_options, list):
            if len(self.solvers_options) != len(self.fragment_solvers):
                raise RuntimeError("The number of solvers options does not match the number of solvers.")

        self.chemical_potential = None
        self.dmet_energy = None

        # Define during the building phase (self.build()).
        self.orbitals = None
        self.orb_list = None
        self.orb_list2 = None

    def build(self):
        # Build adequate mean-field (RHF for now, others in future).
        if not self.mean_field:
            self.mean_field = prepare_mf_RHF(self.molecule)

        # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
        if type(self.electron_localization) == Localization:
            if self.electron_localization == Localization.meta_lowdin:
                self.electron_localization = meta_lowdin_localization
            elif self.electron_localization == Localization.iao:
                self.electron_localization = iao_localization
            else:
                raise ValueError(f"Unsupported ansatz. Built-in localization methods:\n\t{self.builtin_localization}")
        elif not callable(self.electron_localization):
            raise TypeError(f"Invalid electron localization function. Expecting a function.")

        # Construct orbital object
        self.orbitals = helpers._orbitals(self.molecule, self.mean_field, range(self.molecule.nao_nr()), self.electron_localization)

        # TODO: remove last argument, combining fragments not supported
        self.orb_list, self.orb_list2, _ = helpers._fragment_constructor(self.molecule, self.fragment_atoms, 0)

    def _build_scf_fragments(self, onerdm_low, chemical_potential):
        
        scf_fragments = list()

        for i, norb in enumerate(self.orb_list):

            t_list = list()
            t_list.append(norb)
            temp_list = self.orb_list2[i]

            # Construct bath orbitals
            bath_orb, e_occupied = helpers._fragment_bath(self.orbitals.mol_full, t_list, temp_list, onerdm_low)

            # Obtain one particle rdm for a fragment
            norb_high, nelec_high, onerdm_high = helpers._fragment_rdm(t_list, bath_orb, e_occupied,
                                                                        self.orbitals.number_active_electrons)

            # Obtain one particle rdm for a fragment
            one_ele, fock, two_ele = self.orbitals.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

            # Construct guess orbitals for fragment SCF calculations
            guess_orbitals = helpers._fragment_guess(t_list, bath_orb, chemical_potential, norb_high, nelec_high,
                                                        self.orbitals.active_fock)

            # Carry out SCF calculation for a fragment
            mf_fragment, fock_frag_copy, mol_frag = helpers._fragment_scf(t_list, two_ele, fock, nelec_high, norb_high,
                                                            guess_orbitals, chemical_potential)

            scf_fragments.append([mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock])

        return scf_fragments

    def simulate(self):
        """Perform DMET single-shot calculation.

        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            fragment_atoms (list): List of number of atoms for each fragment (int).
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
            fragment_solvers (list): Specifies what solvers should be used for each fragment
                which to solve each fragment. If None is passed here, a defaulot solover is used instead for all fragments.

        Return:
            float64: The DMET energy (dmet_energy).
        """
        # TODO : find a better initial guess than 0.0 for chemical potential. DMET fails often currently.
        # Initialize the energy list and SCF procedure employing newton-raphson algorithm
        self.n_iter = 0
        self.chemical_potential = scipy.optimize.newton(self._oneshot_loop, self.initial_chemical_potential, tol=1e-5)

        if self.verbose:
            print(' \t*** DMET Cycle Done *** ')
            print(' \tDMET Energy ( a.u. ) = ' + '{:17.10f}'.format(self.dmet_energy))
            print(' \tChemical Potential   = ' + '{:17.10f}'.format(self.chemical_potential))

        return self.dmet_energy

    def _oneshot_loop(self, chemical_potential):
        """Perform the DMET loop.

        This is the function which runs in the minimizer.
        DMET calculation converges when the chemical potential is below the
        threshold value of the Newton-Rhapson optimizer.

        Args:
            chemical_potential (float64): The Chemical potential.
            orbitals (numpy.array): The localized orbitals (float64).
            orb_list (list): The number of orbitals for each fragment (int).
            orb_list2 (list): List of lists of the minimum and maximum orbital label for each fragment (int).
            energy_list (list): List of DMET energy for each iteration (float64).
            solvers (list): List of ElectronicStructureSolvers used to solve
                each fragment.

        Returns:
            float64: The new chemical potential.
        """

        # Calculate the 1-RDM for the entire molecule.
        onerdm_low = helpers._low_rdm(self.orbitals.active_fock, self.orbitals.number_active_electrons)

        self.n_iter += 1
        if self.verbose:
            print(" \tIteration = ", self.n_iter)
            print(' \t----------------')
            print(' ')

        number_of_electron = 0.0
        energy_temp = 0.0

        # Carry out SCF calculation for all fragments.
        scf_fragments = self._build_scf_fragments(onerdm_low, chemical_potential)

        for i, info_fragment in enumerate(scf_fragments):
            mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock = info_fragment

            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print('\t\t------------------------')

            # Input shouldnt solver objects, but strings/enum "vqe", "fci", "ccsd", with some options
            # Solver objects should be built on the fly, using the fragment molecule / mean-field as input
            solver_fragment = self.fragment_solvers[i]
            solver_options = self.solvers_options[i]
            if solver_fragment == 'fci':
                solver_fragment = FCISolver()
                solver_fragment.simulate(mol_frag, mf_fragment, **solver_options)
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == 'ccsd':
                solver_fragment = CCSDSolver()
                solver_fragment.simulate(mol_frag, mf_fragment, **solver_options)
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == 'vqe':
                system = {"molecule": mol_frag, "mean_field": mf_fragment}
                solver_fragment = VQESolver({**system, **solver_options})
                solver_fragment.build()
                solver_fragment.simulate()
                onerdm, twordm = solver_fragment.get_rdm(solver_fragment.optimal_var_params)

            fragment_energy, _, one_rdm = self._compute_energy(mf_fragment, onerdm, twordm,
                                                                              fock_frag_copy, t_list, one_ele, two_ele,
                                                                              fock)

            # Sum up the energy
            energy_temp += fragment_energy

            # Sum up the number of electrons
            number_of_electron += np.trace(one_rdm[: t_list[0], : t_list[0]])

            if self.verbose:
                print("\t\tFragment Energy                 = " + '{:17.10f}'.format(fragment_energy))
                print("\t\tNumber of Electrons in Fragment = " + '{:17.10f}'.format(np.trace(one_rdm)))
                print('')

        energy_temp += self.orbitals.core_constant_energy
        self.dmet_energy = energy_temp

        return number_of_electron - self.orbitals.number_active_electrons

    def _compute_energy(self, mf_frag, onerdm, twordm, fock_frag_copy, t_list, oneint, twoint, fock):
        """Calculate the fragment energy.

        Args:
            mf_frag (pyscf.scf.RHF): The mean field of the fragment.
            onerdm (numpy.array): one-particle reduced density matrix (float64).
            twordm (numpy.array): two-particle reduced density matrix (float64).
            fock_frag_copy (numpy.array): Fock matrix with the chemical potential subtracted (float64).
            t_list (list): List of number of fragment and bath orbitals (int).
            oneint (numpy.array): One-electron integrals of fragment (float64).
            twoint (numpy.array): Two-electron integrals of fragment (float64).
            fock (numpy.array): Fock matrix of fragment (float64).

        Returns:
            float64: Fragment energy (fragment_energy).
            float64: Total energy for fragment using RDMs (total_energy_rdm).
            numpy.array: One-particle RDM for a fragment (one_rdm, float64).
        """

        # Execute CCSD calculation
        norb = t_list[0]

        # Calculate the one- and two- RDM for DMET energy calculation (Transform to AO basis)
        one_rdm = reduce(np.dot, (mf_frag.mo_coeff, onerdm, mf_frag.mo_coeff.T))

        twordm = np.einsum('pi,ijkl->pjkl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('qj,pjkl->pqkl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('rk,pqkl->pqrl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('sl,pqrl->pqrs', mf_frag.mo_coeff, twordm)

        # Calculate the total energy based on RDMs
        total_energy_rdm = np.einsum('ij,ij->', fock_frag_copy, one_rdm) + 0.5 * np.einsum('ijkl,ijkl->', twoint,
                                                                                           twordm)

        # Calculate fragment expectation value
        fragment_energy_one_rdm = 0.25 * np.einsum('ij,ij->', one_rdm[: norb, :], fock[: norb, :] + oneint[: norb, :]) \
                                  + 0.25 * np.einsum('ij,ij->', one_rdm[:, : norb], fock[:, : norb] + oneint[:, : norb])

        fragment_energy_twordm = 0.125 * np.einsum('ijkl,ijkl->', twordm[: norb, :, :, :], twoint[: norb, :, :, :]) \
                                 + 0.125 * np.einsum('ijkl,ijkl->', twordm[:, : norb, :, :], twoint[:, : norb, :, :]) \
                                 + 0.125 * np.einsum('ijkl,ijkl->', twordm[:, :, : norb, :], twoint[:, :, : norb, :]) \
                                 + 0.125 * np.einsum('ijkl,ijkl->', twordm[:, :, :, : norb], twoint[:, :, :, : norb])

        fragment_energy = fragment_energy_one_rdm + fragment_energy_twordm

        return fragment_energy, total_energy_rdm, one_rdm
