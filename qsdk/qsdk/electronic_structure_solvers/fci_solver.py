""" Define electronic structure solver employing the full configuration interaction (CI) method """

from pyscf import ao2mo, fci

from qsdk.electronic_structure_solvers.electronic_structure_solver import ElectronicStructureSolver


class FCISolver(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem, through pyscf.

        Args:
            molecule (SecondQuantizedMolecule): The molecule to simulate.

        Attributes:
            ci (numpy.array): The CI wavefunction (float64).
            norb (int): The number of molecular orbitals.
            nelec (int): The number of electrons.
            cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
            mean_field (pyscf.scf): Mean field object.
    """

    def __init__(self, molecule):

        if molecule.get_frozen_orbitals() is not None:
            raise NotImplementedError(f"Frozen orbitals are not implemented in {self.__class__.__name__}.")

        self.ci = None
        self.norb = molecule.n_active_mos
        self.nelec = molecule.n_active_electrons
        self.cisolver = fci.direct_spin0.FCI(molecule.to_pyscf(molecule.basis))
        self.cisolver.verbose = 0
        self.mean_field = molecule.mean_field

    def simulate(self):
        """ Perform the simulation (energy calculation) for the molecule.

            Returns:
                energy (float): Total FCI energy.
        """

        h1 = self.mean_field.mo_coeff.T @ self.mean_field.get_hcore() @ self.mean_field.mo_coeff

        twoint = self.mean_field._eri

        eri = ao2mo.restore(8, twoint, self.norb)
        eri = ao2mo.incore.full(eri, self.mean_field.mo_coeff)
        eri = ao2mo.restore(1, eri, self.norb)

        ecore = self.mean_field.energy_nuc()

        energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], self.nelec, ecore=ecore)

        return energy

    def get_rdm(self):
        """ Compute the Full CI 1- and 2-particle reduced density matrices.

            Returns:
                one_rdm, two_rdm (numpy.array, numpy.array): One & two-particle RDMs
            Raises:
                RuntimeError: If method "simulate" hasn't been run.
        """

        # Check if Full CI is performed
        if self.ci is None:
            raise RuntimeError("FCISolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        one_rdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
        two_rdm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)

        return one_rdm, two_rdm
