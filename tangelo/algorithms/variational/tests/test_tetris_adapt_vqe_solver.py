# Copyright 2023 Good Chemistry Company.
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
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g_uhf_a1_frozen
from tangelo.toolboxes.ansatz_generator._unitary_majorana_cc import get_majorana_uccgsd_pool


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

    def test_multiple_cycle_tetris_adapt_uhf(self):
        """Try running TETRISADAPTSolver with JKMN mapping and uhf H4 with majorana uccgsd pool for 7 iterations"""

        opt_dict = {"molecule": mol_H4_sto3g_uhf_a1_frozen, "max_cycles": 7, "verbose": False,
                    "pool": get_majorana_uccgsd_pool, "pool_args": {"molecule": mol_H4_sto3g_uhf_a1_frozen},
                    "qubit_mapping": "JKMN"}
        adapt_solver = TETRISADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.95831, places=3)
        self.assertTrue(adapt_solver.ansatz.n_var_params > 7)


if __name__ == "__main__":
    unittest.main()
