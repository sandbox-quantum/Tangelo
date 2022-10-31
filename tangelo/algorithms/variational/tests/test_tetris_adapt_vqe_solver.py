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

from tangelo.algorithms.variational import TETRISADAPTSolver
from tangelo.molecule_library import mol_H2_sto3g


class TETRISADAPTSolverTest(unittest.TestCase):

    def test_build_tetris_adapt(self):
        """Try instantiating TETRISADAPTSolver with basic input."""

        opt_dict = {"molecule": mol_H2_sto3g, "max_cycles": 15}
        adapt_solver = TETRISADAPTSolver(opt_dict)
        adapt_solver.build()

    def test_single_cycle_tetris_adapt(self):
        """Try instantiating TETRISADAPTSolver with basic input."""

        opt_dict = {"molecule": mol_H2_sto3g, "max_cycles": 1, "verbose": False}
        adapt_solver = TETRISADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.13727, places=4)


if __name__ == "__main__":
    unittest.main()
