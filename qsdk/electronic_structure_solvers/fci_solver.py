""" Define electronic structure solver employing the full configuration interaction (CI) method """

from pyscf import ao2mo, fci

from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from .electronic_structure_solver import ElectronicStructureSolver


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class FCISolver(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem, through pyscf.

    Attributes:
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
    """

    def __init__(self):
        self.ci = None
        self.norb = None
        self.nelec = None
        self.cisolver = None

    def simulate(self, molecule, mean_field=None):
        """ Perform the simulation (energy calculation) for the molecule.

        If the mean field is not provided it is automatically calculated.
        `pyscf.ao2mo` is used to transform the AO integrals into MO integrals.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
        Returns:
            float64: The Full CI energy
        """

        # Calculate the mean field if the user has not already done it
        if not mean_field:
            mean_field = prepare_mf_RHF(molecule)

        h1 = mean_field.mo_coeff.T @ mean_field.get_hcore() @ mean_field.mo_coeff
        twoint = mean_field._eri
        self.norb = len(mean_field.mo_energy)
        eri = ao2mo.restore(8, twoint, self.norb)
        eri = ao2mo.incore.full(eri, mean_field.mo_coeff)
        eri = ao2mo.restore(1, eri, self.norb)
        self.cisolver = fci.direct_spin0.FCI(molecule)
        self.cisolver.verbose = 0
        self.nelec = molecule.nelectron
        energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], self.nelec, ecore=molecule.energy_nuc())

        return energy

    def get_rdm(self):
        """ Compute the Full CI 1- and 2-particle reduce density matrices.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (fci_onerdm & fci_twordm, float64).
        Raises:
            RuntimeError: If no simulation has been run.
        """        

        # Check if Full CI is performed
        if not self.norb or not self.nelec:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        fci_onerdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
        fci_twordm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)

        return fci_onerdm, fci_twordm
