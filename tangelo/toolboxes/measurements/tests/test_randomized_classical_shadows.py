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

import numpy as np

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.measurements import RandomizedClassicalShadow
from tangelo.toolboxes.operators import QubitOperator

# Circuit to sample (Bell state).
state = Circuit([Gate("H", 0), Gate("CNOT", 1, 0)])

# Simple saved bistrings and unitaries to construct a shadow (size = 100).
bitstrings = ["01", "00", "00", "01", "00", "00", "11", "11", "00", "11", "11",
              "01", "00", "11", "11", "00", "11", "11", "00", "11", "11", "10",
              "01", "11", "01", "00", "00", "01", "01", "11", "01", "01", "11",
              "10", "00", "01", "01", "00", "00", "10", "10", "10", "00", "11",
              "00", "01", "00", "11", "00", "00", "11", "11", "11", "00", "10",
              "01", "01", "10", "01", "00", "00", "10", "00", "00", "10", "10",
              "01", "00", "11", "01", "00", "11", "11", "00", "11", "11", "01",
              "01", "01", "11", "00", "11", "10", "11", "10", "00", "00", "00",
              "00", "10", "01", "10", "10", "11", "11", "11", "01", "00", "11",
              "11"]
unitaries = ["ZY", "XZ", "XZ", "YZ", "ZX", "XX", "YZ", "XZ", "YX", "ZZ", "XX",
             "YY", "XX", "XX", "ZZ", "YZ", "XX", "XZ", "XY", "YX", "XZ", "ZY",
             "YX", "XY", "YZ", "XX", "XX", "ZY", "ZY", "XZ", "YY", "XZ", "YX",
             "YY", "ZZ", "YY", "YZ", "ZX", "XY", "XY", "YY", "YY", "XZ", "YZ",
             "XZ", "XZ", "ZZ", "XX", "XZ", "ZX", "ZZ", "ZX", "ZZ", "XX", "YX",
             "ZX", "XY", "YY", "YY", "XX", "YX", "YZ", "XX", "ZZ", "XZ", "YY",
             "YX", "ZY", "XZ", "ZX", "XX", "XX", "YX", "XY", "ZZ", "XZ", "ZX",
             "ZY", "YX", "ZZ", "XX", "ZY", "YZ", "ZX", "YZ", "ZY", "XZ", "XX",
             "XZ", "YZ", "YY", "ZY", "XZ", "XX", "ZY", "XX", "YY", "XY", "ZX",
             "ZX"]


class RandomizedClassicalShadowTest(unittest.TestCase):

    def test_initialization(self):
        """Testing the initialization."""

        RandomizedClassicalShadow(state, bitstrings, unitaries)

    def test_same_length_check(self):
        """Testing the case where arguments are not the same length."""

        wrong_bitstrings = bitstrings + ["00"]
        with self.assertRaises(AssertionError):
            RandomizedClassicalShadow(state, wrong_bitstrings, unitaries)

    def test_shadow_properties(self):
        """Testing of the shadow properties."""

        cs = RandomizedClassicalShadow(state, bitstrings, unitaries, shuffle=False)

        self.assertEqual(cs.n_qubits, 2)
        self.assertEqual(cs.size, 100)
        self.assertEqual(len(cs), 100)

    def test_get_term_observable(self):
        """Testing the computation of a single qubit term."""

        cs = RandomizedClassicalShadow(state, bitstrings, unitaries, shuffle=False)
        obs = cs.get_term_observable([(0, "Y"), (1, "Y")], 1., k=10)
        self.assertAlmostEqual(obs, -0.89999, places=4)

    def test_get_observable(self):
        """Testings the computation of an eigenvalue of a QubitOperator."""

        cs = RandomizedClassicalShadow(state, bitstrings, unitaries, shuffle=False)
        obs = cs.get_observable(QubitOperator("Y0 Y1", coefficient=1.))
        self.assertAlmostEqual(obs, -0.89999, places=4)

    def test_estimate_state(self):
        """Testing of the state estimation method to get the density matrix."""

        cs = RandomizedClassicalShadow(state, bitstrings, unitaries, shuffle=False)
        rho_estimate = cs.estimate_state()

        # Previously ran with this specific shadow.
        rho_ref = np.array([
            [ 0.445+0.j, 0.06+0.045j, 0.255+0.09j, 0.63-0.09j],
            [ 0.06-0.045j, 0.07+0.j, 0.135+0.045j, -0.15-0.09j],
            [ 0.255-0.09j, 0.135-0.045j, 0.025+0.j, -0.12-0.045j],
            [ 0.63+0.09j, -0.15+0.09j, -0.12+0.045j, 0.46+0.j]
        ])

        np.testing.assert_array_almost_equal(rho_estimate, rho_ref)

    def test_get_basis_circuits(self):
        """Testing of the method to get the appended circuit corresponding to
        the unitaries.
        """
        cs = RandomizedClassicalShadow(state, bitstrings, unitaries)

        self.assertEqual(len(cs.get_basis_circuits(False)), 100)
        self.assertEqual(len(cs.get_basis_circuits(True)), 9)


if __name__ == "__main__":
    unittest.main()
