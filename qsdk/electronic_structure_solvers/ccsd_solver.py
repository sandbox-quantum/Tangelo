""" Class performing electronic structure calculation employing the CCSD method """

import warnings
from pyscf import cc, scf

from .electronic_structure_solver import ElectronicStructureSolver


class CCSDSolver(ElectronicStructureSolver):
    """ Uses the CCSD method to solve the electronic structure problem, through pyscf

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
    """
    
    def __init__(self):
        self.cc_fragment = None

    def simulate(self, molecule, mean_field=None):
        """Perform the simulation (energy calculation) for the molecule.
        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.

        Returns:
            float64: CCSD energy (total_energy).
        """

        # TODO: all of this mean-field code needs to be a function call to our "mean field calculation" module.
        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()
        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("CCSDSolver simulating with mean field not converged.", RuntimeWarning)

        # Execute CCSD calculation
        self.cc_fragment = cc.ccsd.CCSD(mean_field)
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
            (numpy.array, numpy.array): One & two-particle RDMs (cc_onerdm & cc_twordm, float64).
        Raises:
            RuntimeError: If no simulation has been run.
        """        

        # Check if CCSD calculation is performed
        if not self.cc_fragment:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        self.cc_fragment.solve_lambda()
        cc_onerdm = self.cc_fragment.make_rdm1()
        cc_twordm = self.cc_fragment.make_rdm2()

        return cc_onerdm, cc_twordm
