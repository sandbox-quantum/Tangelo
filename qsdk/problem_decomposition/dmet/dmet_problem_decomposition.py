import warnings

import scipy
from pyscf import scf
from functools import reduce
import numpy as np

from qsdk.electronic_structure_solvers import FCISolver, CCSDSolver, VQESolver
from ..problem_decomposition import ProblemDecomposition
from ..electron_localization import iao_localization
from . import _helpers as helpers


class DMETProblemDecomposition(ProblemDecomposition):
    """Employ DMET as a problem decomposition technique.

    DMET single-shot algorithm is used for problem decomposition technique.
    By default, CCSD is used as the electronic structure solver, and
    IAO is used for the localization scheme.
    Users can define other electronic structure solver such as FCI or
    VQE as an impurity solver. Meta-Lowdin localization scheme can be
    used instead of the IAO scheme, which cannot be used for minimal
    basis set.

    Attributes:
        electronic_structure_solver (subclass of ElectronicStructureSolver): A type of electronic structure solver. Default is CCSD.
        electron_localization_method (string): A type of localization scheme. Default is IAO.
    """

    def __init__(self):
        self.verbose = True
        self.default_electronic_structure_solver = 'ccsd'
        self.electron_localization_method = iao_localization

    def simulate(self, molecule, fragment_atoms, mean_field=None, fragment_solvers=None):
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

        Raise:
            RuntimeError: If the sum of the atoms in the fragments is different
                from the number of atoms in the molecule.
            RuntimeError: If the number fragments is different from the number
                of solvers.
        """

        # Check if the number of fragment sites is equal to the number of atoms in the molecule
        if molecule.natm != sum(fragment_atoms):
            raise RuntimeError("The number of fragment sites is not equal to the number of atoms in the molecule")

        # Check that the number of solvers matches the number of fragments.
        if fragment_solvers:
            if len(fragment_solvers) != len(fragment_atoms):
                raise RuntimeError("The number of solvers does not match the number of fragments.")

        # TODO : hide this under wrapper to our mean-field calculation module
        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()
        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("DMET simulating with mean field not converged.", RuntimeWarning)

        # Construct orbital object
        orbitals = helpers._orbitals(molecule, mean_field, range(molecule.nao_nr()), self.electron_localization_method)

        # TODO: remove last argument, combining fragments not supported
        orb_list, orb_list2, atom_list2 = helpers._fragment_constructor(molecule, fragment_atoms, 0)

        # TODO : find a better initial guess than 0.0 for chemical potential. DMET fails often currently.
        # Initialize the energy list and SCF procedure employing newton-raphson algorithm
        energy = []
        chemical_potential = 0.0
        chemical_potential = scipy.optimize.newton(self._oneshot_loop, chemical_potential,
                                                   args=(orbitals, orb_list, orb_list2, energy, fragment_solvers),
                                                   tol=1e-5)

        # Get the final energy value
        niter = len(energy)
        dmet_energy = energy[niter - 1]

        if self.verbose:
            print(' \t*** DMET Cycle Done *** ')
            print(' \tDMET Energy ( a.u. ) = ' + '{:17.10f}'.format(dmet_energy))
            print(' \tChemical Potential   = ' + '{:17.10f}'.format(chemical_potential))

        return dmet_energy

    def _oneshot_loop(self, chemical_potential, orbitals, orb_list, orb_list2, energy_list, solvers=None):
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

        # Calculate the 1-RDM for the entire molecule
        onerdm_low = helpers._low_rdm(orbitals.active_fock, orbitals.number_active_electrons)

        niter = len(energy_list) + 1

        if self.verbose:
            print(" \tIteration = ", niter)
            print(' \t----------------')
            print(' ')

        number_of_electron = 0.0
        energy_temp = 0.0

        for i, norb in enumerate(orb_list):
            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print('\t\t------------------------')

            t_list = []
            t_list.append(norb)
            temp_list = orb_list2[i]

            # Construct bath orbitals
            bath_orb, e_occupied = helpers._fragment_bath(orbitals.mol_full, t_list, temp_list, onerdm_low)

            # Obtain one particle rdm for a fragment
            norb_high, nelec_high, onerdm_high = helpers._fragment_rdm(t_list, bath_orb, e_occupied,
                                                                       orbitals.number_active_electrons)

            # Obtain one particle rdm for a fragment
            one_ele, fock, two_ele = orbitals.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

            # Construct guess orbitals for fragment SCF calculations
            guess_orbitals = helpers._fragment_guess(t_list, bath_orb, chemical_potential, norb_high, nelec_high,
                                                     orbitals.active_fock)

            # Carry out SCF calculation for a fragment
            mf_fragment, fock_frag_copy, mol_frag = helpers._fragment_scf(t_list, two_ele, fock, nelec_high, norb_high,
                                                                          guess_orbitals, chemical_potential)

            # Input shouldnt solver objects, but strings/enum "vqe", "fci", "ccsd", with some options
            # Solver objects should be built on the fly, using the fragment molecule / mean-field as input
            solver_fragment = self.default_electronic_structure_solver if not solvers else solvers[i]
            if solver_fragment == 'fci':
                solver_fragment = FCISolver()
                energy = solver_fragment.simulate(mol_frag, mf_fragment)
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == 'ccsd':
                solver_fragment = CCSDSolver()
                energy = solver_fragment.simulate(mol_frag, mf_fragment)
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == 'vqe':
                solver_fragment = VQESolver({"molecule": mol_frag, "mean_field": mf_fragment, "verbose": True,
                                             "initial_var_params": "ones"})
                solver_fragment.build()
                energy = solver_fragment.simulate()
                onerdm, twordm = solver_fragment.get_rdm(solver_fragment.optimal_var_params)


            fragment_energy, total_energy_rdm, one_rdm = self._compute_energy(mf_fragment, onerdm, twordm,
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

        energy_temp += orbitals.core_constant_energy
        energy_list.append(energy_temp)

        return number_of_electron - orbitals.number_active_electrons

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
