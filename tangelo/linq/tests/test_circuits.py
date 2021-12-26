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

"""
    A test class to check that the circuit class is working properly, and processing the information
    of the gate objects it holds is as expected.
"""

import unittest
import copy
from math import pi
from collections import Counter
from tangelo.linq import Gate, Circuit

# Create several abstract circuits with different features
mygates = list()
mygates.append(Gate("H", 2))
mygates.append(Gate("CNOT", 1, control=0))
mygates.append(Gate("CNOT", 2, control=1))
mygates.append(Gate("Y", 0))
mygates.append(Gate("RX", 1, parameter=2.))

circuit1 = Circuit()
for gate in mygates:
    circuit1.add_gate(gate)
circuit2 = Circuit(mygates)

circuit3 = Circuit(mygates)
circuit3.add_gate(Gate("RZ", 4, parameter="some angle", is_variational=True))

circuit4 = copy.deepcopy(circuit3)
circuit4.add_gate(Gate("RY", 3, parameter="some angle", is_variational=True))


class TestCircuits(unittest.TestCase):

    def test_init(self):
        """ Tests that constructing a circuit consisting of several instructions works through either passing
        a list of gates directly , or by using the add_gate method repeatedly on an existing circuit. Simply test
        that the number of gates is what was expected """

        self.assertTrue(circuit1.size == len(mygates))
        self.assertTrue(circuit2.size == len(mygates))

    def test_is_variational(self):
        """ Ensure that the circuit is labeled as variational as soon as one variational gate is present """

        self.assertTrue(circuit1.is_variational is False)
        self.assertTrue(circuit3.is_variational is True)

    def test_width(self):
        """ Ensure the width attribute of the circuit object (number of qubits) matches the gate operations
        present in the circuit. """

        self.assertTrue(circuit1.width == 3)
        self.assertTrue(circuit3.width == 5)

    def test_gate_counts(self):
        """ Test that all gates have been counted """

        n_gates = sum(circuit1._gate_counts.values())
        self.assertTrue(circuit1.size == n_gates)

    def test_add_circuits(self):
        """ Test the concatenation of two circuit objects """

        circuit_sum = circuit3 + circuit4
        self.assertTrue(circuit_sum.size == circuit3.size + circuit4.size)
        self.assertTrue(circuit_sum.is_variational == circuit3.is_variational or circuit4.is_variational)
        self.assertTrue(circuit_sum._qubit_indices == circuit3._qubit_indices.union(circuit4._qubit_indices))
        self.assertTrue(circuit_sum._gate_counts == dict(Counter(circuit3._gate_counts)
                                                         + Counter(circuit4._gate_counts)))
        self.assertTrue(len(circuit_sum._variational_gates) == (len(circuit3._variational_gates) +
                                                                len(circuit4._variational_gates)))

    def test_mul_circuit(self):
        """ Test the multiplication (repetition) operator for circuit objects """

        # Should work for *2
        c2 = circuit3 * 2
        ref_counts = {k: 2*v for k, v in circuit3._gate_counts.items()}
        self.assertTrue(c2._gate_counts == ref_counts)
        self.assertTrue(len(c2._variational_gates) == 2*len(circuit3._variational_gates))

        # Fail for incorrect values, such as 2.5, or 0.
        with self.assertRaises(ValueError):
            _ = circuit3 * 2.5
        with self.assertRaises(ValueError):
            _ = circuit3 * 0

        # Repeating an empty circuit yields an empty circuit
        self.assertTrue((Circuit()*3).size == 0)

        # Check on right-hand side
        self.assertTrue(2*circuit3 == c2)

    def test_equality_circuit(self):
        """ Test equality operators (== and !=) for circuits """
        self.assertTrue(circuit1 == circuit2)
        self.assertTrue(circuit3 != circuit2)
        c3 = Circuit(circuit3._gates, n_qubits=6)
        self.assertTrue(circuit3 != c3)

    def test_fixed_sized_circuit_above(self):
        """ If circuit is instantiated with fixed width, the code must throw if qubit indices are not consistent """
        circuit_fixed = Circuit(n_qubits=2)
        gate1 = Gate("H", 2)
        gate2 = Gate("CNOT", 0, control=5)
        self.assertRaises(ValueError, circuit_fixed.add_gate, gate1)
        self.assertRaises(ValueError, circuit_fixed.add_gate, gate2)

    def test_fixed_sized_circuit_below(self):
        """ If circuit is instantiated with fixed width, then the width property must be consistent, regardless
            of what gate instructions have been passed """
        n_qubits = 3
        circuit_fixed = Circuit([Gate("H", 0)], n_qubits=n_qubits)
        self.assertTrue(circuit_fixed.width == n_qubits)

    def test_inverse(self):
        """ Test if inverse function returns the proper set of gates by comparing print strings."""

        mygates_inverse = list()
        mygates_inverse.append(Gate("RX", 1, parameter=-2.))
        mygates_inverse.append(Gate("Y", 0))
        mygates_inverse.append(Gate("CNOT", 2, control=1))
        mygates_inverse.append(Gate("CNOT", 1, control=0))
        mygates_inverse.append(Gate("H", 2))
        circuit1_inverse = Circuit(mygates_inverse)
        self.assertTrue(circuit1.inverse(), circuit1_inverse)

        ts_circuit = Circuit([Gate("T", 0), Gate("S", 1)])
        ts_circuit_inverse = Circuit([Gate("PHASE", 0, parameter=-pi/4), Gate("PHASE", 0, parameter=-pi/2)])
        self.assertTrue(ts_circuit.inverse(), ts_circuit_inverse)


if __name__ == "__main__":
    unittest.main()
