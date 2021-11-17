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

from tangelo.algorithms.classical.ccsd_solver import CCSDSolver
from tangelo.molecule_library import mol_H2_321g, mol_Be_321g


# TODO: Can we test the get_rdm method on H2 ? How do we get our reference? Whole matrix or its properties?
class CCSDSolverTest(unittest.TestCase):

    def test_ccsd_h2(self):
        """Test CCSDSolver against result from reference implementation."""

        solver = CCSDSolver(mol_H2_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.1478300596229851, places=6)

    def test_ccsd_be(self):
        """Test CCSDSolver against result from reference implementation."""

        solver = CCSDSolver(mol_Be_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.531416589890926, places=6)

    def test_ccsd_be_frozen_core(self):
        """ Test CCSDSolver against result from reference implementation, with
        no mean-field provided as input. Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = CCSDSolver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.530687987160581, places=6)

    def test_ccsd_be_as_two_levels(self):
        """ Test CCSDSolver against result from reference implementation, with
        no mean-field provided as input. This atom is reduced to an HOMO-LUMO
        problem.
        """

        mol_Be_321g_freeze_list = mol_Be_321g.freeze_mos([0, 3, 4, 5, 6, 7, 8], inplace=False)

        solver = CCSDSolver(mol_Be_321g_freeze_list)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.498104489160106, places=6)

    def test_ccsd_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without
        first running a simulation.
        """

        solver = CCSDSolver(mol_H2_321g)
        self.assertRaises(RuntimeError, solver.get_rdm)


if __name__ == "__main__":
    unittest.main()
