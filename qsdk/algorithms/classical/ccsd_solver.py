"""Class performing electronic structure calculation employing the CCSD method.
"""

from pyscf import cc

from qsdk.algorithms.electronic_structure_solver import ElectronicStructureSolver


class CCSDSolver(ElectronicStructureSolver):
    """Uses the CCSD method to solve the electronic structure problem, through
    pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
        mean_field (pyscf.scf.RHF): The mean field of the molecule.
        frozen (list or int): Frozen molecular orbitals.
    """

    def __init__(self, molecule):
        self.cc_fragment = None

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

            Returns:
                total_energy (float): CCSD energy
        """
        # Execute CCSD calculation
        self.cc_fragment = cc.CCSD(self.mean_field, frozen=self.frozen)
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7

        correlation_energy, _, _ = self.cc_fragment.ccsd()
        total_energy = self.mean_field.e_tot + correlation_energy

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices. The CCSD
        lambda equation will be solved for calculating the RDMs.

            Returns:
                one_rdm, two_rdm (numpy.array, numpy.array): One & two-particle
                    RDMs
            Raises:
                RuntimeError: If no simulation has been run.
        """

        # Check if CCSD calculation is performed
        if self.cc_fragment is None:
            raise RuntimeError("CCSDSolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        self.cc_fragment.solve_lambda()
        one_rdm = self.cc_fragment.make_rdm1()
        two_rdm = self.cc_fragment.make_rdm2()

        return one_rdm, two_rdm
