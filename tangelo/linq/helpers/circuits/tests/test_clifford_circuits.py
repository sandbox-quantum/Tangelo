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

from tangelo.linq import Gate, Circuit
from tangelo.linq.helpers.circuits.clifford_circuits import decompose_gate_to_cliffords
from tangelo.linq.translator.translate_cirq import translate_c_to_cirq

non_clifford_gate = Gate("RY", 0, parameter=np.pi/3)

class CliffordCircuitTest(unittest.TestCase):

    def test_decompose_gate_to_cliffords(self):
        """Test if gate decomposition returns correct sequence"""
        for gate in ["RY", "RX", "RZ", "PHASE"]:
            for param in [np.pi / 2, np.pi, np.pi / 2]:
                ref_gate = Gate(gate, 0, parameter=param)
                u_ref = translate_c_to_cirq(Circuit([ref_gate]))
                u_decompose = translate_c_to_cirq(Circuit(decompose_gate_to_cliffords(ref_gate)))
                np.testing.assert_array_almost_equal(u_ref.unitary(), u_decompose.unitary())

    def test_non_clifford_gates(self):
        """Test if non-clifford gate raises value error"""

        self.assertRaises(ValueError, decompose_gate_to_cliffords, non_clifford_gate)

