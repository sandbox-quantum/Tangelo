# Copyright 2021 1QB Information Technologies Inc.
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

"""Tests for statevector mapping methods, which carry a numpy array indicating
fermionic occupation of reference state into qubit representation.
"""

import unittest
import numpy as np

from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector, vector_to_circuit


class TestVector(unittest.TestCase):

    def test_jw_value(self):
        """Check that Jordan-Wigner mapping returns correct vector, for both
        default spin orderings.
        """
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_updown = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        output_jw = get_vector(vector.size, sum(vector), mapping="jw", up_then_down=False)
        output_jw_updown = get_vector(vector.size, sum(vector), mapping="jw", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector - output_jw), 0.0)
        self.assertEqual(np.linalg.norm(vector_updown - output_jw_updown), 0.0)

    def test_bk_value(self):
        """Check that Bravyi-Kitaev mapping returns correct vector, for both
        default spin orderings.
        """
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_bk = np.array([1, 0, 1, 0, 0, 0, 0, 0])
        vector_bk_updown = np.array([1, 0, 0, 0, 1, 0, 0, 0])

        output_bk = get_vector(vector.size, sum(vector), mapping="bk", up_then_down=False)
        output_bk_updown = get_vector(vector.size, sum(vector), mapping="bk", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector_bk - output_bk), 0.0)
        self.assertEqual(np.linalg.norm(vector_bk_updown - output_bk_updown), 0.0)

    def test_scbk_value(self):
        """Check that symmetry-conserving Bravyi-Kitaev mapping returns correct
        vector.
        """
        vector = np.array([1, 0, 0, 1, 0, 0])

        output_bk = get_vector(8, 4, mapping="SCBK", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector - output_bk), 0.0)

    def test_circuit_build(self):
        """Check circuit width and size (number of X gates)."""
        vector = np.array([1, 1, 1, 1, 0, 0, 1, 1])
        circuit = vector_to_circuit(vector)
        self.assertEqual(circuit.size, sum(vector))
        self.assertEqual(circuit.width, vector.size)


if __name__ == "__main__":
    unittest.main()
