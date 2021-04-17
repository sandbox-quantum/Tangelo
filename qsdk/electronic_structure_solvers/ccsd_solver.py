""" Class performing electronic structure calculation employing the CCSD method """

from pyscf import cc

from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from .electronic_structure_solver import ElectronicStructureSolver


class CCSDSolver(ElectronicStructureSolver):
    """ Uses the CCSD method to solve the electronic structure problem, through pyscf

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
    """
    
    def __init__(self):
        self.cc_fragment = None

    def simulate(self, molecule, mean_field=None, frozen_orbitals=None):
        """Perform the simulation (energy calculation) for the molecule.
        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
            frozen_orbitals (int or list of int): Frozen orbitals (int -> first 
                n are forzen, list -> indexes are frozen).

        Returns:
            total_energy (float): CCSD energy
        """

        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = prepare_mf_RHF(molecule)

        # If an empty list is passed, it is converted to None (for PySCF).
        if frozen_orbitals == []:
            frozen_orbitals = None

        # Execute CCSD calculation
        self.cc_fragment = cc.ccsd.CCSD(mean_field, frozen=frozen_orbitals)
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7
        correlation_energy, t1, t2 = self.cc_fragment.ccsd()
        scf_energy = mean_field.e_tot
        total_energy = scf_energy + correlation_energy

        return total_energy

    def get_rdm(self):
        """ Calculate the 1- and 2-particle reduced density matrices. The CCSD lambda equation will be solved for
         calculating the RDMs.

        Returns:
            one_rdm, two_rdm (numpy.array, numpy.array): One & two-particle RDMs
        Raises:
            RuntimeError: If no simulation has been run.
        """        

        # Check if CCSD calculation is performed
        if not self.cc_fragment:
            raise RuntimeError("CCSDSolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        self.cc_fragment.solve_lambda()
        one_rdm = self.cc_fragment.make_rdm1()
        two_rdm = self.cc_fragment.make_rdm2()

        return one_rdm, two_rdm
