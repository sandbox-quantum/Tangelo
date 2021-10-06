import unittest

from qsdk.algorithms import FCISolver
from qsdk.molecule_library import mol_H2_321g, mol_Be_321g, mol_H4_cation_sto3g


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class FCISolverTest(unittest.TestCase):

    def test_fci_h2(self):
        """Test FCISolver against result from reference implementation (H2)."""

        solver = FCISolver(mol_H2_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.1478300596229851, places=6)

    def test_fci_be(self):
        """Test FCISolver against result from reference implementation (Be)."""

        solver = FCISolver(mol_Be_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.531444379108095, places=6)

    def test_fci_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without
        first running a simulation.
        """

        solver = FCISolver(mol_H2_321g)
        self.assertRaises(RuntimeError, solver.get_rdm)

    def test_fci_openshell(self):
        """Test that the fci implementation for an openshell molecule is working properly"""

        solver = FCISolver(mol_H4_cation_sto3g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.6419373, places=6)

    def test_fci_be_frozen_core(self):
        """ Test FCISolver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = FCISolver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.530687987160581, places=6)


if __name__ == "__main__":
    unittest.main()
