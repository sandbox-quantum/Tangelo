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
from tangelo.toolboxes.measurements import DerandomizedClassicalShadow
from tangelo.toolboxes.operators import QubitOperator

# Circuit to sample (Bell state).
state = Circuit([Gate("H", 0), Gate("CNOT", 1, 0)])

# Simple saved bistrings and unitaries to construct a shadow (size = 100).
bitstrings = ["11", "01", "11", "01", "00", "10", "00", "10", "11", "01", "00",
              "01", "00", "01", "11", "10", "00", "01", "11", "01", "00", "01",
              "11", "01", "11", "01", "00", "01", "11", "10", "11", "10", "11",
              "10", "00", "10", "00", "10", "00", "01", "11", "10", "11", "10",
              "00", "01", "11", "01", "11", "10", "00", "10", "00", "01", "11",
              "10", "00", "10", "11", "10", "11", "10", "00", "01", "11", "01",
              "11", "01", "11", "10", "00", "01", "11", "01", "11", "10", "11",
              "01", "11", "10", "11", "10", "00", "10", "11", "01", "00", "10",
              "00", "10", "11", "10", "11", "01", "11", "01", "00", "10", "11",
              "10"]
unitaries = ["XX", "YY"] * 50


class DerandomizedClassicalShadowTest(unittest.TestCase):

    def test_initialization(self):
        """Testing the initialization."""

        DerandomizedClassicalShadow(state, bitstrings, unitaries)

    def test_same_length_check(self):
        """Testing the case where arguments are not the same length."""

        wrong_bitstrings = bitstrings + ["00"]
        with self.assertRaises(AssertionError):
            DerandomizedClassicalShadow(state, wrong_bitstrings, unitaries)

    def test_shadow_properties(self):
        """Testing of the shadow properties."""

        cs = DerandomizedClassicalShadow(state, bitstrings, unitaries)

        self.assertEqual(cs.n_qubits, 2)
        self.assertEqual(cs.size, 100)
        self.assertEqual(len(cs), 100)

    def test_get_term_observable(self):
        """Testing the computation of a single qubit term."""

        cs = DerandomizedClassicalShadow(state, bitstrings, unitaries)

        obs_xx = cs.get_term_observable([(0, "X"), (1, "X")], 1.)
        self.assertAlmostEqual(obs_xx, 1.0, places=4)

        obs_yy = cs.get_term_observable([(0, "Y"), (1, "Y")], 1.)
        self.assertAlmostEqual(obs_yy, -1.0, places=4)

    def test_get_observable(self):
        """Testings the computation of an eigenvalue of a QubitOperator."""

        cs = DerandomizedClassicalShadow(state, bitstrings, unitaries)
        obs_xx = cs.get_observable(QubitOperator("X0 X1", coefficient=1.))
        self.assertAlmostEqual(obs_xx, 1.0, places=4)

        obs_yy = cs.get_observable(QubitOperator("Y0 Y1", coefficient=1.))
        self.assertAlmostEqual(obs_yy, -1.0, places=4)

    def test_get_basis_circuits(self):
        """Testing of the method to get the appended circuit corresponding to
        the unitaries.
        """
        cs = DerandomizedClassicalShadow(state, bitstrings, unitaries)

        self.assertEqual(len(cs.get_basis_circuits(False)), 100)
        self.assertEqual(len(cs.get_basis_circuits(True)), 2)


if __name__ == "__main__":
    unittest.main()
