import unittest

from qsdk.electronic_structure_solvers.ccsd_solver import CCSDSolver
from qsdk import SecondQuantizedMolecule

H2 = """
   H 0.00 0.00 0.0
   H 0.00 0.00 0.74137727
   """

Be = """Be 0.0 0.0 0.0"""


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class CCSDSolverTest(unittest.TestCase):

    def test_ccsd_h2(self):
        """ Test CCSDSolver against result from reference implementation. """

        mol = SecondQuantizedMolecule(H2, basis="3-21g", frozen_orbitals=None)

        solver = CCSDSolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.1478300596229851, places=8)

    def test_ccsd_be(self):
        """ Test CCSDSolver against result from reference implementation. """

        mol = SecondQuantizedMolecule(Be, basis="3-21g", frozen_orbitals=None)

        solver = CCSDSolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.531416589890926, places=8)

    def test_ccsd_be_frozen_core(self):
        """ Test CCSDSolver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol = SecondQuantizedMolecule(Be, basis="3-21g", frozen_orbitals=1)

        solver = CCSDSolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.530687987160581, places=8)

    def test_ccsd_be_as_two_levels(self):
        """ Test CCSDSolver against result from reference implementation, with no mean-field provided as input.
            This atom is reduced to an HOMO-LUMO problem.
        """

        mol = SecondQuantizedMolecule(Be, basis="3-21g", frozen_orbitals=[0, 3, 4, 5, 6, 7, 8])

        solver = CCSDSolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.498104489160106, places=8)

    def test_ccsd_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without first running a simulation."""
        mol = SecondQuantizedMolecule(H2, basis="3-21g", frozen_orbitals=None)
        solver = CCSDSolver(mol)
        self.assertRaises(RuntimeError, solver.get_rdm)


if __name__ == "__main__":
    unittest.main()
