# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from tangelo.molecule_library import mol_pyridine
from tangelo.algorithms.classical.semi_empirical_solver import MINDO3Solver


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
