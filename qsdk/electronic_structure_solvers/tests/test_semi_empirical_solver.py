import unittest

from qsdk.molecule_library import mol_pyridine
from qsdk.electronic_structure_solvers.semi_empirical_solver import MINDO3Solver


class MINDO3SolverTest(unittest.TestCase):

    def test_energy(self):
        """Test MINDO3Solver with pyridine. Validated with:
            - MINDO/3-derived geometries and energies of alkylpyridines and the
            related N-methylpyridinium cations. Jeffrey I. Seeman,
            John C. Schug, and Jimmy W. Viers
            The Journal of Organic Chemistry 1983 48 (14), 2399-2407
            DOI: 10.1021/jo00162a021.
        """

        solver = MINDO3Solver(mol_pyridine)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -33.04112644117467, places=6)


if __name__ == "__main__":
    unittest.main()
