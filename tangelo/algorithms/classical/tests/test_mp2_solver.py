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

import numpy as np

from tangelo.algorithms.classical.mp2_solver import MP2Solver, default_mp2_solver
from tangelo.molecule_library import mol_H2_321g, mol_Be_321g, mol_H2_sto3g, mol_H2_sto3g_uhf


class MP2SolverTest(unittest.TestCase):

    def test_h2(self):
        """Test MP2Solver against result from reference implementation (H2)."""

        solver = MP2Solver(mol_H2_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.14025452, places=3)

    @unittest.skipIf("pyscf" != default_mp2_solver, "Test Skipped: Only functions for pyscf \n")
    def test_be(self):
        """Test MP2Solver against result from reference implementation (Be)."""

        solver = MP2Solver(mol_Be_321g)
        energy = solver.simulate()
        self.assertAlmostEqual(energy, -14.51026131, places=3)

        # Assert energy calculated from RDMs and MP2 calculation are the same.
        one_rdm, two_rdm = solver.get_rdm()
        self.assertAlmostEqual(mol_Be_321g.energy_from_rdms(one_rdm, two_rdm), energy)

    def test_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without
        first running a simulation.
        """

        solver = MP2Solver(mol_H2_321g)
        self.assertRaises(RuntimeError, solver.get_rdm)

    def test_be_frozen_core(self):
        """ Test MP2Solver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = MP2Solver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.5092873, places=3)

    @unittest.skipIf("pyscf" != default_mp2_solver, "Test Skipped: Only functions for pyscf \n")
    def test_get_mp2_params_restricted(self):
        """Test the packing of RMP2 amplitudes as initial parameters for coupled
        cluster based methods.
        """

        solver = MP2Solver(mol_H2_sto3g)
        solver.simulate()

        ref_params = [2.e-05, 3.632537e-02]

        np.testing.assert_array_almost_equal(ref_params, solver.get_mp2_amplitudes())

    @unittest.skipIf("pyscf" != default_mp2_solver, "Test Skipped: Only functions for pyscf \n")
    def test_get_mp2_params_unrestricted(self):
        """Test the packing of UMP2 amplitudes as initial parameters for coupled
        cluster based methods.
        """

        solver = MP2Solver(mol_H2_sto3g_uhf)
        solver.simulate()
        ref_params = [0., 0., 0.030736]

        np.testing.assert_array_almost_equal(ref_params, solver.get_mp2_amplitudes())

    @unittest.skipIf("pyscf" != default_mp2_solver, "Test Skipped: Only functions for pyscf \n")
    def test_get_mp2_params_restricted(self):
        """Test the packing of RMP2 amplitudes as initial parameters for coupled
        cluster based methods.
        """

        solver = MP2Solver(mol_H2_sto3g)
        solver.simulate()

        ref_params = [2.e-05, 3.632537e-02]

        np.testing.assert_array_almost_equal(ref_params, solver.get_mp2_amplitudes())

    @unittest.skipIf("pyscf" != default_mp2_solver, "Test Skipped: Only functions for pyscf \n")
    def test_get_mp2_params_unrestricted(self):
        """Test the packing of UMP2 amplitudes as initial parameters for coupled
        cluster based methods.
        """

        solver = MP2Solver(mol_H2_sto3g_uhf)
        solver.simulate()
        ref_params = [0., 0., 0.030736]

        np.testing.assert_array_almost_equal(ref_params, solver.get_mp2_amplitudes())


if __name__ == "__main__":
    unittest.main()
