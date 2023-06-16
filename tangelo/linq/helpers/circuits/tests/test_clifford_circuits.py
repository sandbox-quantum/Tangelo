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

from math import pi
from tangelo.linq import Gate

from tangelo.linq.helpers.circuits.clifford_circuits import decompose_gate_to_cliffords

clifford_gate = Gate("RX", target=0, parameter=pi / 2)
non_clifford_gate = Gate("RZ", target=0, parameter=0.2)
clifford_decomposed = [Gate("SDAG", 0), Gate("H", 0), Gate("SDAG", 0)]


class CliffordCircuitTest(unittest.TestCase):

    def test_decompose_gate_to_cliffords(self):
        """Test if gate decomposition returns correct sequence"""

        self.assertEqual(clifford_decomposed, decompose_gate_to_cliffords(clifford_gate))

    def test_non_clifford_gates(self):
        """Test if non-clifford gate raises value error"""

        self.assertRaises(ValueError, decompose_gate_to_cliffords, non_clifford_gate)

