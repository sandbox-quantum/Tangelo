"""Employ DMET as a problem decomposition technique. """

from enum import Enum
from functools import reduce
from pyscf import scf
import scipy
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
    """DMET single-shot algorithm is used for problem decomposition technique.
    By default, CCSD is used as the electronic structure solver, and
    Meta-Lowdin is used for the localization scheme.
    Users can define other electronic structure solver such as FCI or
    VQE as an impurity solver. IAO scheme can be used instead of the Meta-Lowdin
    localization scheme, but it cannot be used for minimal basis set.

    Attributes:
        molecule (pyscf.gto.mol): The molecular system.
        mean-field (optional): Mean-field of molecular system.
        electron_localization (Localization): A type of localization scheme. Default is Meta-Lowdin.
        fragment_atoms (list): List of number of atoms in each fragment. Sum of this list
            should be the same as the number of atoms in the original system.
        fargment_solvers (list or str): List of solvers for each fragment. If only a string is
            detected, this solver is used for all fragments.
        optimizer (function handle): A function defining the classical optimizer and its behavior.
        initial_chemical_potential (str or array-like) : Initial value for the chemical potential.
        solvers_options (list or dict): List of dictionaries for the solver options. If only a single
            dictionary is passed, the same options are applied for every solver. This will raise an error
            if different solver are parsed.
        verbose (bool) : Flag for DMET verbosity.
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
                           "optimizer": self._default_optimizer,
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
        # If there is no options, all default ones are applied.
        # If a single options dictionary is parsed, it is repeated for every solvers.
        # If it is a list, we verified that the length is the same.
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

        # Results of the DMET loops.
        self.chemical_potential = None
        self.dmet_energy = None

        # Define during the building phase (self.build()).
        self.orbitals = None
        self.orb_list = None
        self.orb_list2 = None
        self.onerdm_low = None

    def build(self):
        """Building the orbitals list for each fragment. It sets the values of 
        self.orbitals, self.orb_list and self.orb_list2.
        """

        # Build adequate mean-field (RHF for now, others in future).
        if not self.mean_field:
            self.mean_field = prepare_mf_RHF(self.molecule)

        # Locate electron with a built-in method. A custom one can be provided.
        if type(self.electron_localization) == Localization:
            if self.electron_localization == Localization.meta_lowdin:
                self.electron_localization = meta_lowdin_localization
            elif self.electron_localization == Localization.iao:
                self.electron_localization = iao_localization
            else:
                raise ValueError(f"Unsupported ansatz. Built-in localization methods:\n\t{self.builtin_localization}")
        elif not callable(self.electron_localization):
            raise TypeError(f"Invalid electron localization function. Expecting a function.")

        # Construct orbital object.
        self.orbitals = helpers._orbitals(self.molecule, self.mean_field, range(self.molecule.nao_nr()), self.electron_localization)

        # TODO: remove last argument, combining fragments not supported.
        self.orb_list, self.orb_list2, _ = helpers._fragment_constructor(self.molecule, self.fragment_atoms, 0)

        # Calculate the 1-RDM for the entire molecule.
        self.onerdm_low = helpers._low_rdm(self.orbitals.active_fock, self.orbitals.number_active_electrons)

    def simulate(self):
        """Perform DMET loop to optimize the chemical potential. It converges
        when the electron summation across all fragments is the same as the 
        number of electron in the molecule.

        Returns:
            float: The DMET energy (dmet_energy).
        """

        # To keep track the number of iteration (was done with an energy list before).
        # TODO: A decorator function to do the same thing?
        self.n_iter = 0

        # Initialize the energy list and SCF procedure employing newton-raphson algorithm.
        # TODO : find a better initial guess than 0.0 for chemical potential. DMET fails often currently.
        if not self.orbitals:
            raise RuntimeError("No fragment built. Have you called DMET.build ?")

        self.chemical_potential = self.optimizer(self._oneshot_loop, self.initial_chemical_potential)

        if self.verbose:
            print(' \t*** DMET Cycle Done *** ')
            print(' \tDMET Energy ( a.u. ) = ' + '{:17.10f}'.format(self.dmet_energy))
            print(' \tChemical Potential   = ' + '{:17.10f}'.format(self.chemical_potential))

        return self.dmet_energy

    def _build_scf_fragments(self, chemical_potential):
        """Building the orbitals list for each fragment. It sets the values of 
        self.orbitals, self.orb_list and self.orb_list2.

        Args:
            onerdm_low (matrix): 1-RDM for the whole molecule.
            chemical_potential (float): Variational parameter for DMET. 

        Returns:
            list: List of many objects important for each fragments.
        """

        # Empty list, all fragment informations will be returned in this list.
        scf_fragments = list()

        for i, norb in enumerate(self.orb_list):
            t_list = list()
            t_list.append(norb)
            temp_list = self.orb_list2[i]

            # Construct bath orbitals.
            bath_orb, e_occupied = helpers._fragment_bath(self.orbitals.mol_full, t_list, temp_list, self.onerdm_low)

            # Obtain one particle rdm for a fragment.
            norb_high, nelec_high, onerdm_high = helpers._fragment_rdm(t_list, bath_orb, e_occupied,
                                                                        self.orbitals.number_active_electrons)

            # Obtain one particle rdm for a fragment.
            one_ele, fock, two_ele = self.orbitals.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

            # Construct guess orbitals for fragment SCF calculations.
            guess_orbitals = helpers._fragment_guess(t_list, bath_orb, chemical_potential, norb_high, nelec_high,
                                                        self.orbitals.active_fock)

            # Carry out SCF calculation for a fragment.
            mf_fragment, fock_frag_copy, mol_frag = helpers._fragment_scf(t_list, two_ele, fock, nelec_high, norb_high,
                                                            guess_orbitals, chemical_potential)

            scf_fragments.append([mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock])

        return scf_fragments

    def _oneshot_loop(self, chemical_potential):
        """Perform the DMET loop. This is the cost function which is optimized
        with respect to the chemical potential.

        Args:
            chemical_potential (float): The chemical potential.

        Returns:
            float: The new chemical potential.
        """

        self.n_iter += 1
        if self.verbose:
            print(" \tIteration = ", self.n_iter)
            print(' \t----------------')
            print(' ')

        number_of_electron = 0.0
        energy_temp = 0.0

        # Carry out SCF calculation for all fragments.
        scf_fragments = self._build_scf_fragments(chemical_potential)

        # Iterate across all fragment and compute their energies.
        # The total energy is stored in energy_temp.
        for i, info_fragment in enumerate(scf_fragments):

            # Unpacking the information for the selected fragment.
            mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock = info_fragment

            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print('\t\t------------------------')

            # TODO: Changing this into something more simple is preferable. There 
            # would be an enum class with every solver in it. After this, we would
            # define every solver in a list and call them recursively.
            # FCISolver and CCSDSolver must be taken care of, but this is a PR itself.
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
                                                               fock_frag_copy, t_list, one_ele, 
                                                               two_ele, fock)

            # Sum up the energy.
            energy_temp += fragment_energy

            # Sum up the number of electrons.
            number_of_electron += np.trace(one_rdm[: t_list[0], : t_list[0]])

            if self.verbose:
                print("\t\tFragment Energy                 = " + '{:17.10f}'.format(fragment_energy))
                print("\t\tNumber of Electrons in Fragment = " + '{:17.10f}'.format(np.trace(one_rdm)))
                print('')

        energy_temp += self.orbitals.core_constant_energy
        self.dmet_energy = energy_temp

        return number_of_electron - self.orbitals.number_active_electrons

    def get_resources(self):
        """ Estimate the resources required by DMET. Only supports fragments solved 
        with VQESolver. Resources for each fragments are outputed as a list.
        """

        # Carry out SCF calculation for all fragments.
        scf_fragments = self._build_scf_fragments(self.initial_chemical_potential)

        # Store ressources for each fragments.
        resources_fragments = [None for _ in range(len(scf_fragments))]

        # Iterate across all fragment and compute their energies.
        # The total energy is stored in energy_temp.
        for i, info_fragment in enumerate(scf_fragments):

            # Unpacking the information for the selected fragment.
            mf_fragment, _, mol_frag, _, _, _, _ = info_fragment

            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print('\t\t------------------------')

            # Buiding SCF fragments and quantum circuit. Resources are then 
            # estimated. For classical sovlers, this functionality is not 
            # implemented yet.
            solver_fragment = self.fragment_solvers[i]
            solver_options = self.solvers_options[i]
            if solver_fragment == 'vqe':
                system = {"molecule": mol_frag, "mean_field": mf_fragment}
                solver_fragment = VQESolver({**system, **solver_options})
                solver_fragment.build()
                vqe_ressources = solver_fragment.get_resources()
                resources_fragments[i] = vqe_ressources
                print("\t\t{}\n".format(vqe_ressources))
            else:
                print("\t\tRessources estimation not supported for {} solver.\n".format(self.fragment_solvers[i]))

        return resources_fragments

    def _compute_energy(self, mf_frag, onerdm, twordm, fock_frag_copy, t_list, oneint, twoint, fock):
        """Calculate the fragment energy.

        Args:
            mf_frag (pyscf.scf.RHF): The mean field of the fragment.
            onerdm (numpy.array): one-particle reduced density matrix (float).
            twordm (numpy.array): two-particle reduced density matrix (float).
            fock_frag_copy (numpy.array): Fock matrix with the chemical potential subtracted (float).
            t_list (list): List of number of fragment and bath orbitals (int).
            oneint (numpy.array): One-electron integrals of fragment (float).
            twoint (numpy.array): Two-electron integrals of fragment (float).
            fock (numpy.array): Fock matrix of fragment (float).

        Returns:
            float: Fragment energy (fragment_energy).
            float: Total energy for fragment using RDMs (total_energy_rdm).
            numpy.array: One-particle RDM for a fragment (one_rdm, float).
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

    def _default_optimizer(self, func, var_params):
        """ Function used as a default optimizer for DMET when user does not provide one.

        Args:
            func (function handle): The function that performs energy estimation.
                This function takes var_params as input and returns a float.
            var_params (list): The variational parameters (float).
        Returns:
            The optimal chemical potential found by the optimizer.
        """

        result = scipy.optimize.newton(func, var_params, tol=1e-5)

        return result
