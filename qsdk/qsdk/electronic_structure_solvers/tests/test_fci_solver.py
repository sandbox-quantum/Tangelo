import unittest

from qsdk.electronic_structure_solvers import FCISolver
from qsdk.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule

H2 = """
    H 0.00 0.00 0.0
    H 0.00 0.00 0.74137727
    """

Be = """Be 0.0 0.0 0.0"""


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class FCISolverTest(unittest.TestCase):

    def test_fci_h2(self):
        """ Test FCISolver against result from reference implementation (H2). """

        mol = SecondQuantizedMolecule(H2, basis="3-21g", frozen_orbitals=None)

        solver = FCISolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.1478300596229851, places=8)

    def test_fci_be(self):
        """ Test FCISolver against result from reference implementation (Be). """

        mol = SecondQuantizedMolecule(Be, basis="3-21g", frozen_orbitals=None)

        solver = FCISolver(mol)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.531444379108095, places=8)

    def test_fci_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without first running a simulation."""

        mol = SecondQuantizedMolecule(H2, basis="3-21g", frozen_orbitals=None)
        solver = FCISolver(mol)
        self.assertRaises(RuntimeError, solver.get_rdm)


if __name__ == "__main__":
    unittest.main()
