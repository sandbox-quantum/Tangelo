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

from tangelo.linq import Circuit, Gate
from tangelo.toolboxes.post_processing import post_select, strip_post_selection, ancilla_symmetry_circuit

circ = Circuit([Gate("H", 0), Gate("CNOT", 1, 0)])

sym_circ = Circuit([Gate("H", 0), Gate("CNOT", 1, 0),
                    Gate('RY', 0, parameter=-1.5707963267948966),
                    Gate("CNOT", 2, 0), Gate("CNOT", 2, 1),
                    Gate('RY', 0, parameter=1.5707963267948966)])

hist = {"000": 0.0087, "001": 0.0003, "010": 0.0056, "011": 0.0481,
        "100": 0.0053, "101": 0.0035, "110": 0.9136, "111": 0.0149}


class PostSelectionTest(unittest.TestCase):

    def test_symmetry_circuit(self):
        """Test the ancilla symmetry circuit constructor"""
        with self.assertRaises(RuntimeError):
            ancilla_symmetry_circuit(circ, "XYZ")

        with self.assertWarns(UserWarning):
            test_circ = ancilla_symmetry_circuit(circ, "X")

        self.assertEqual(sym_circ, test_circ)

    def test_post_select(self):
        """Test equality of post-selected frequencies with reference values rounded to 1e-4"""
        hist_ref = {'00': 0.0093, '01': 0.0060, '10': 0.0057, '11': 0.9790}
        hist_post = post_select(hist, {2: "0"})

        for key, value in hist_ref.items():
            self.assertAlmostEqual(value, hist_post[key], delta=1e-4)

    def test_strip_post_selection(self):
        """Test stripping of ancilla and aggregation of corresponding frequencies"""
        hist_ref = {"00": 0.0090, "01": 0.0537, "10": 0.0088, "11": 0.9285}
        hist_strip = strip_post_selection(hist, 2)

        self.assertDictEqual(hist_ref, hist_strip)


if __name__ == "__main__":
    unittest.main()
