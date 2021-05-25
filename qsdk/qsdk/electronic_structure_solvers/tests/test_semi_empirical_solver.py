import unittest
from pyscf import gto

from qsdk.electronic_structure_solvers.semi_empirical_solver import MINDO3Solver

pyridine = """
  C      1.3603      0.0256      0.0000
  C      0.6971     -1.2020      0.0000
  C     -0.6944     -1.2184      0.0000
  C     -1.3895     -0.0129      0.0000
  C     -0.6712      1.1834      0.0000
  N      0.6816      1.1960      0.0000
  H      2.4530      0.1083      0.0000
  H      1.2665     -2.1365      0.0000
  H     -1.2365     -2.1696      0.0000
  H     -2.4837      0.0011      0.0000
  H     -1.1569      2.1657      0.0000
"""


class MINDO3SolverTest(unittest.TestCase):

    def test_energy(self):
        """ Test MINDO3Solver with pyridine. Validated with:

        MINDO/3-derived geometries and energies of alkylpyridines and the related N-methylpyridinium cations
        Jeffrey I. Seeman, John C. Schug, and Jimmy W. Viers
        The Journal of Organic Chemistry 1983 48 (14), 2399-2407
        DOI: 10.1021/jo00162a021
        """

        mol = gto.Mole()
        mol.atom =  pyridine
        mol.charge = 0
        mol.spin = 0
        mol.build()

        solver = MINDO3Solver()
        energy = solver.simulate(mol)

        self.assertAlmostEqual(energy, -33.04112644117467, places=6)


if __name__ == "__main__":
    unittest.main()
