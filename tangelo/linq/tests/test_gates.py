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

from tangelo.linq import Gate


class TestGates(unittest.TestCase):
    """
        A test class to check that the circuit class is working properly, and processing the information
        of the gate objects it holds as expected.
    """

    def test_some_gates(self):
        """ Test that some basic gates can be invoked with a few different parameters, and that this information
        is printed as expected by the built-in __str__ method """

        # Create a Hadamard gate acting on qubit 2
        H_gate = Gate("H", 2)
        # Create a CNOT gate with control qubit 0 and target qubit 1
        CNOT_gate = Gate("CNOT", 1, 0)
        # Create a parameterized rotation on qubit 1 with angle 2 radians
        RX_gate = Gate("RX", 1, parameter=2.)
        # Create a parameterized rotation on qubit 1 , with an undefined angle, that will be variational
        RZ_gate = Gate("RZ", 1, parameter="an expression", is_variational=True)
        # Create a multi-controlled X gate with a numpy array
        CCCX_gate = Gate("CX", 0, control=np.array([1, 2, 4], dtype=np.int32))

        for gate in [H_gate, CNOT_gate, RX_gate, RZ_gate, CCCX_gate]:
            print(gate)

    def test_repr(self):
        """ Test that some basic gates can be invoked with a few different parameters, and that this information
        is printed as expected by the built-in __repr__ method end eval(gate.__repr__()) returns the same gate."""

        # Create a Hadamard gate acting on qubit 2
        H_gate = Gate("H", 2)
        # Create a CNOT gate with control qubit 0 and target qubit 1
        CNOT_gate = Gate("CNOT", 1, 0)
        # Create a parameterized rotation on qubit 1 with angle 2 radians
        RX_gate = Gate("RX", 1, parameter=2.)
        # Create a parameterized rotation on qubit 1 , with an undefined angle, that will be variational
        RZ_gate = Gate("RZ", 1, parameter="an expression", is_variational=True)
        # Create a multi-controlled X gate with a numpy array
        CCCX_gate = Gate("CX", 0, control=np.array([1, 2, 4], dtype=np.int32))

        for gate in [H_gate, CNOT_gate, RX_gate, RZ_gate, CCCX_gate]:
            self.assertEqual(eval(gate.__repr__()), gate)

    def test_some_gates_inverse(self):
        """ Test that some basic gates can be inverted with a few different parameters, and fails when non-invertible
        parameters are passed"""

        # Create a Hadamard gate acting on qubit 2
        H_gate = Gate("H", 2)
        H_gate_inverse = Gate("H", 2)
        self.assertEqual(H_gate.inverse(), H_gate_inverse)

        # Create a SWAP gate acting on qubits 1 and 2
        swap_gate = Gate("SWAP", [1, 2])
        swap_gate_inverse = Gate("SWAP", [1, 2])
        self.assertEqual(swap_gate.inverse(), swap_gate_inverse)

        # Create a parameterized rotation on qubit 1 with angle 2 radians
        RX_gate = Gate("RX", 1, parameter=2.)
        RX_gate_inverse = Gate("RX", 1, parameter=-2.)
        self.assertEqual(RX_gate.inverse(), RX_gate_inverse)

        # Create a parameterized rotation on qubit 1 , with an undefined angle, that will be variational
        RZ_gate = Gate("RZ", 1, parameter="an expression", is_variational=True)
        with self.assertRaises(AttributeError):
            RZ_gate.inverse()

    def test_is_clifford(self):
        """ Test that some basic gates are correctly identified as Clifford or non Clifford"""

        # test single qubit Clifford gates
        for name in {"H", "S", "X", "Z", "Y", "SDAG"}:
            self.assertEqual(True, Gate(name, 0).is_clifford())

        # test two qubit Clifford gates
        for name in {"CNOT", "CX", "CY", "CZ"}:
            self.assertEqual(True, Gate(name, target=0, control=1).is_clifford())

        # test parameterized single qubit gates at Clifford and non Clifford parameters
        for name in {"RX", "RY", "RZ", "PHASE"}:
            for clifford_point in [0, np.pi/2, np.pi, -np.pi/2, 4*np.pi]:
                self.assertEqual(True, Gate(name, 0, parameter=clifford_point).is_clifford())
            for non_clifford_point in [0.1, -np.pi/3, 5*np.pi/4, 1.5]:
                self.assertEqual(False, Gate(name, 0, parameter=non_clifford_point).is_clifford())

        # test two qubit non Clifford gates
        for name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            self.assertEqual(False, Gate(name, target=0, control=1).is_clifford())

    def test_incorrect_gate(self):
        """ Test to catch a gate with inputs that do not make sense """

        self.assertRaises(ValueError, Gate, "H", -1)
        self.assertRaises(ValueError, Gate, "CNOT", 0, control=0.3)
        self.assertRaises(ValueError, Gate, 'X', target=0, control=1)
        self.assertRaises(ValueError, Gate, "CNOT", target=0, control=0)

    def test_integer_types(self):
        """ Test to catch error with incorrect target or control qubit index type"""
        self.assertRaises(ValueError, Gate, "CSWAP", target=[0, 'a'], control=np.array([1], dtype=np.int32))
        self.assertRaises(ValueError, Gate, "X", target=0, control=[-1, 2, 3],)

    def test_gate_equality(self):
        """ Test behaviour of == and != operators on gates """
        g1 = Gate("CPOTATO", target=2, control=0, parameter=0, is_variational=True)
        g2 = Gate("CPOTATO", target=2, control=0, parameter="", is_variational=True)
        g3 = Gate("CPOTATO", target=2, control=0, parameter=0, is_variational=True)

        self.assertTrue(g1 == g3)
        self.assertTrue(g1 != g2)

    def test_gate_equality_modulo_twopi(self):
        """ Test behaviour of == with parameter outside [0, 2pi]. """
        g1 = Gate("POTATO", target=0, parameter=np.pi, is_variational=True)
        g2 = Gate("POTATO", target=0, parameter=3*np.pi, is_variational=True)
        g3 = Gate("POTATO", target=0, parameter=-np.pi, is_variational=True)

        self.assertTrue(g1 == g2)
        self.assertTrue(g1 == g3)
        self.assertTrue(g2 == g3)

    def test_too_many_qubits_on_gates(self):
        """ Test the behavior when too many qubits are selected for a gate. """

        # Try to create a Hadamard gate acting on qubits 0 and 1.
        self.assertRaises(ValueError, Gate, "H", target=[0, 1])

        # Try to create a XX gate acting on qubits 0, 1 and 2.
        self.assertRaises(ValueError, Gate, "XX", target=[0, 1, 2])

    def test_non_hermitian_gates_inverse(self):
        """ Test that non-hermitian gates (S, T) can be inversed."""

        S_gate = Gate("S", 0)
        S_gate_inverse = Gate("PHASE", 0, parameter=-np.pi/2)
        self.assertEqual(S_gate.inverse(), S_gate_inverse)

        T_gate = Gate("T", 0)
        T_gate_inverse = Gate("PHASE", 0, parameter=-np.pi/4)
        self.assertEqual(T_gate.inverse(), T_gate_inverse)


if __name__ == "__main__":
    unittest.main()
