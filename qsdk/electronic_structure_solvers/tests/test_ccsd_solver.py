import unittest
from pyscf import gto, scf

from ..ccsd_solver import CCSDSolver

H2 = """
   H 0.00 0.00 0.0
   H 0.00 0.00 0.74137727
   """

Be = """Be 0.0 0.0 0.0"""


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class CCSDSolverTest(unittest.TestCase):

    def test_ccsd_h2_no_mf(self):
        """ Test CCSDSolver against result from reference implementation, with no mean-field provided as input """
        mol = gto.Mole()
        mol.atom = H2
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        solver = CCSDSolver()
        energy = solver.simulate(mol)

        self.assertAlmostEqual(energy, -1.1478300596229851, places=8)

    def test_ccsd_h2_with_mf(self):
        """ Test CCSDSolver against result from reference implementation, with mean-field provided as input """
        mol = gto.Mole()
        mol.atom = H2
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()

        solver = CCSDSolver()
        energy = solver.simulate(mol, mf)

        self.assertAlmostEqual(energy, -1.1478300596229851, places=8)

    def test_ccsd_be_no_mf(self):
        """ Test CCSDSolver against result from reference implementation, with no mean-field provided as input """

        mol = gto.Mole()
        mol.atom = Be
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        solver = CCSDSolver()
        energy = solver.simulate(mol)

        self.assertAlmostEqual(energy, -14.531416589890926, places=8)

    def test_ccsd_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without first running a simulation."""
        solver = CCSDSolver()
        self.assertRaises(RuntimeError, solver.get_rdm)


if __name__ == "__main__":
    unittest.main()
