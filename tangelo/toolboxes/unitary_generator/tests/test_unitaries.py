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
from tangelo.toolboxes.unitary_generator import TrotterSuzukiUnitary, CircuitUnitary
from tangelo.toolboxes.operators import QubitOperator


# UCC1 hard coding circuit. Simple example not relying on import Ansatz.
circuit = Circuit([Gate("CNOT", 1, 0), Gate("RZ", 1, parameter=0.1, is_variational=True), Gate("CNOT", 1, 0)])
qu_op = QubitOperator("Z0 Z1")


class UnitariesTest(unittest.TestCase):

    def test_init_circuit_unitary(self):
        """Test initialization of the CircuitUnitary class."""
        CircuitUnitary(circuit)

        with self.assertRaises(ValueError):
            CircuitUnitary(circuit, "not_valid")

    def test_build_circuit_circuit_unitary(self):
        """Test building controlled circuit for CircuitUnitary."""

        circuit_unitary = CircuitUnitary(circuit)
        ccircuit = circuit_unitary.build_circuit(1, 2, "all")
        self.assertEqual(ccircuit, Circuit([Gate("CNOT", 1, [0, 2]), Gate("CRZ", 1, 2, parameter=0.1, is_variational=True),
                                            Gate("CNOT", 1, [0, 2])]))

        ccircuit = circuit_unitary.build_circuit(1, 2, "variational")
        self.assertEqual(ccircuit, Circuit([Gate("CNOT", 1, 0), Gate("CRZ", 1, 2, parameter=0.1, is_variational=True), Gate("CNOT", 1, 0)]))

        with self.assertRaises(ValueError):
            circuit_unitary.build_circuit(1, 2, "not_valid")

    def test_init_trotter_suzuki(self):
        """Test initialization of the TrotterSuzukiUnitary class."""

        TrotterSuzukiUnitary(qu_op, 0.1)

        with self.assertRaises(ValueError):
            TrotterSuzukiUnitary(qu_op, n_steps_method="not_valid")

    def test_build_circuit_trotter_suzuki_unitary(self):
        """Test building of controlled circuit for TrotterSuzukiUnitary"""

        trotter_unitary = TrotterSuzukiUnitary(qu_op, time=1)

        ccircuit = trotter_unitary.build_circuit(2, 2, "time")
        self.assertEqual(ccircuit, Circuit([Gate("CNOT", 1, 0), Gate("CRZ", 1, 2, parameter=4.0), Gate("CNOT", 1, 0)]))

        ccircuit = trotter_unitary.build_circuit(2, 2, "repeat")
        self.assertEqual(ccircuit, Circuit([Gate("CNOT", 1, 0), Gate("CRZ", 1, 2, parameter=2.0), Gate("CNOT", 1, 0)])*2)

        with self.assertRaises(ValueError):
            trotter_unitary.build_circuit(2, 2, "not_valid")


if __name__ == "__main__":
    unittest.main()
