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

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.measurements import AdaptiveClassicalShadow
from tangelo.toolboxes.operators import QubitOperator

# Circuit to sample (Bell state).
state = Circuit([Gate("H", 0), Gate("CNOT", 1, 0)])

# Simple saved bistrings and unitaries to construct a shadow (size = 100).
bitstrings = ["10", "00", "11", "00", "00", "11", "10", "11", "00", "00", "11",
              "00", "00", "01", "00", "11", "11", "00", "11", "11", "11", "00",
              "11", "11", "01", "00", "11", "01", "11", "11", "01", "10", "00",
              "11", "11", "11", "11", "11", "10", "01", "00", "00", "11", "01",
              "00", "11", "11", "00", "11", "11", "11", "00", "11", "10", "11",
              "11", "00", "11", "10", "01", "11", "10", "10", "01", "00", "10",
              "10", "11", "11", "00", "10", "11", "00", "11", "01", "01", "01",
              "11", "10", "00", "00", "11", "00", "01", "10", "01", "00", "01",
              "00", "01", "11", "00", "00", "00", "10", "11", "00", "11", "10",
              "11"]
unitaries = ["YY", "XX", "ZZ", "XX", "ZZ", "ZZ", "YY", "XX", "XX", "XX", "ZZ",
             "ZZ", "XX", "YY", "ZZ", "ZZ", "XX", "XX", "ZZ", "XX", "ZZ", "ZZ",
             "XX", "ZZ", "YY", "XX", "ZZ", "YY", "XX", "XX", "YY", "YY", "ZZ",
             "XX", "XX", "ZZ", "ZZ", "ZZ", "YY", "YY", "ZZ", "XX", "XX", "YY",
             "XX", "XX", "XX", "ZZ", "XX", "XX", "XX", "ZZ", "XX", "YY", "ZZ",
             "ZZ", "XX", "ZZ", "YY", "YY", "XX", "YY", "YY", "YY", "XX", "YY",
             "YY", "XX", "XX", "ZZ", "YY", "ZZ", "ZZ", "ZZ", "YY", "YY", "YY",
             "ZZ", "YY", "ZZ", "ZZ", "XX", "ZZ", "YY", "YY", "YY", "XX", "YY",
             "ZZ", "YY", "ZZ", "XX", "XX", "XX", "YY", "XX", "ZZ", "XX", "YY",
             "XX"]


class AdaptiveClassicalShadowTest(unittest.TestCase):

    def test_initialization(self):
        """Testing the initialization."""

        AdaptiveClassicalShadow(state, bitstrings, unitaries)

    def test_same_length_check(self):
        """Testing the case where arguments are not the same length."""

        wrong_bitstrings = bitstrings + ["00"]
        with self.assertRaises(AssertionError):
            AdaptiveClassicalShadow(state, wrong_bitstrings, unitaries)

    def test_shadow_properties(self):
        """Testing of the shadow properties."""

        cs = AdaptiveClassicalShadow(state, bitstrings, unitaries)

        self.assertEqual(cs.n_qubits, 2)
        self.assertEqual(cs.size, 100)
        self.assertEqual(len(cs), 100)

    def test_get_term_observable(self):
        """Testing the computation of a single qubit term."""

        cs = AdaptiveClassicalShadow(state, bitstrings, unitaries)

        obs_xx = cs.get_term_observable([(0, "X"), (1, "X")], 1.)
        self.assertAlmostEqual(obs_xx, 1.0, places=4)

        obs_yy = cs.get_term_observable([(0, "Y"), (1, "Y")], 1.)
        self.assertAlmostEqual(obs_yy, -1.0, places=4)

    def test_get_observable(self):
        """Testings the computation of an eigenvalue of a QubitOperator."""

        cs = AdaptiveClassicalShadow(state, bitstrings, unitaries)
        obs_xx = cs.get_observable(QubitOperator("X0 X1", coefficient=1.))
        self.assertAlmostEqual(obs_xx, 1.0, places=4)

        obs_yy = cs.get_observable(QubitOperator("Y0 Y1", coefficient=1.))
        self.assertAlmostEqual(obs_yy, -1.0, places=4)

    def test_get_basis_circuits(self):
        """Testing of the method to get the appended circuit corresponding to
        the unitaries.
        """
        cs = AdaptiveClassicalShadow(state, bitstrings, unitaries)

        self.assertEqual(len(cs.get_basis_circuits(False)), 100)
        self.assertEqual(len(cs.get_basis_circuits(True)), 3)


if __name__ == "__main__":
    unittest.main()
