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

from tangelo.linq import Gate, Circuit, stack

# Create several abstract circuits with different features
mygates = [Gate("H", 2), Gate("CNOT", 1, control=0), Gate("CNOT", 2, control=1),
           Gate("Y", 0), Gate("RX", 1, parameter=2.)]

circuit1 = Circuit()
for gate in mygates:
    circuit1.add_gate(gate)
circuit2 = Circuit(mygates)

circuit3 = Circuit(mygates)
circuit3.add_gate(Gate("RZ", 4, parameter="some angle", is_variational=True))

circuit4 = copy.deepcopy(circuit3)
circuit4.add_gate(Gate("RY", 3, parameter="some angle", is_variational=True))

entangle_circuit = Circuit([Gate("CSWAP", target=[2, 5], control=[0]),
                            Gate("CSWAP", target=[3, 7], control=[4]),
                            Gate("H", 6)], n_qubits=10)


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

    def test_gate_data_is_copied(self):
        """ Ensure that circuit is not referencing mutable variables that could cause it to change after
        instantiation if the values of the variables are later changed in external code. """

        mygates2 = copy.deepcopy(mygates)
        c1 = Circuit(mygates2)

        g = mygates2[0]
        g.target.append(1)
        g.name = 'POTATO'
        g.parameter = -999.

        c2 = Circuit(mygates2)
        self.assertTrue(c1 != c2)

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

    def test_entangled_indices(self):
        """ Test that entangled indices subsets are properly updated after
        a new gate is added to the circuit. """

        c = Circuit()
        c.add_gate(Gate("CNOT", target=4, control=0))
        self.assertTrue(c.get_entangled_indices() == [{0, 4}])
        c.add_gate(Gate("CNOT", target=5, control=1))
        self.assertTrue(c.get_entangled_indices() == [{0, 4}, {1, 5}])
        c.add_gate(Gate("H", target=2))
        self.assertTrue(c.get_entangled_indices() == [{0, 4}, {1, 5}, {2}])
        c.add_gate(Gate("CNOT", target=6, control=5))
        self.assertTrue(c.get_entangled_indices() == [{0, 4}, {2}, {1, 5, 6}])
        c.add_gate(Gate("CNOT", target=1, control=4))
        self.assertTrue(c.get_entangled_indices() == [{2}, {0, 1, 4, 5, 6}])
        c.add_gate(Gate("CSWAP", target=[2, 7], control=[0]))
        self.assertTrue(c.get_entangled_indices() == [{0, 1, 2, 4, 5, 6, 7}])

    def test_trim_circuit(self):
        """ Check that unnecessary indices are trimmed and new indices minimal """

        ref_c = Circuit([Gate("CSWAP", target=[1, 4], control=[0]),
                         Gate("CSWAP", target=[2, 6], control=[3]),
                         Gate("H", 5)], n_qubits=7)
        entangle_circuit.trim_qubits()
        self.assertTrue(ref_c == entangle_circuit)

    def test_reindex_qubits(self):
        """ Test the function that reindexes qubits (e.g replaces indices by another). """

        # With circuit of natural width
        gates = [Gate("H", 2), Gate("CNOT", 1, control=0), Gate("CSWAP", target=[1, 2], control=[0])]
        c1 = Circuit(copy.deepcopy(gates))
        c1.reindex_qubits([4, 5, 6])

        ref = [Gate("H", 6), Gate("CNOT", 5, control=4), Gate("CSWAP", target=[5, 6], control=[4])]
        self.assertTrue(ref == c1._gates)

        # With circuit of fixed width (sends 4,5,6 to 0,1,2, the rest is not relevant)
        c2 = Circuit(ref, n_qubits=8)
        c2.reindex_qubits([3, 4, 5, 6, 0, 1, 2, 7])
        self.assertTrue(gates == c2._gates)

        # Test for input of incorrect length in both previous cases
        with self.assertRaises(ValueError):
            c1.reindex_qubits([2])
        with self.assertRaises(ValueError):
            c2.reindex_qubits([0, 1, 2])

    def test_split_circuit(self):
        """ Test function that splits circuit into several circuits targeting qubit subsets
        that are not entangled with each other. Trims unnecessary qubit indices. """
        c = Circuit([Gate("CSWAP", target=[2, 5], control=[0]),
                     Gate("CSWAP", target=[3, 7], control=[4]),
                     Gate("H", 6)])
        c1, c2, c3 = c.split()

        self.assertTrue(c1 == Circuit([Gate("CSWAP", target=[1, 2], control=[0])]))
        self.assertTrue(c2 == Circuit([Gate("CSWAP", target=[0, 2], control=[1])]))
        self.assertTrue(c3 == Circuit([Gate("H", target=0)]))

    def test_stack_circuits(self):
        """ Test circuit stacking """

        c1 = Circuit([Gate("H", 6)])
        c2 = Circuit([Gate("CNOT", 5, control=4)])
        c3 = Circuit([Gate("CSWAP", target=[5, 6], control=[4])])

        ref = [Gate("H", 0), Gate("CNOT", 2, control=1), Gate("CSWAP", target=[4, 5], control=[3])]

        # No and multiple arguments, natural or as an unpacked list
        self.assertTrue(ref == stack(c1, c2, c3)._gates)
        self.assertTrue(ref == stack(*[c1, c2, c3])._gates)
        self.assertTrue([] == stack(*[])._gates)

        # Try convenience method in Circuit class
        self.assertTrue(ref == c1.stack(c2, c3)._gates)

        c4 = Circuit([Gate("H", 0), Gate("CNOT", 1, control=0), Gate("X", 0), Gate("RX", 1, parameter=2.)])

        ref2 = [Gate("H", 0), Gate("CNOT", 1, control=0), Gate("X", 0), Gate("RX", 1, parameter=2.),
                Gate("H", 2), Gate("CNOT", 3, control=2), Gate("X", 2), Gate("RX", 3, parameter=2.)]
        # Stacked copies of same circuit
        self.assertTrue(ref2 == stack(c4, c4)._gates)
        self.assertTrue(ref2 == c4.stack(c4)._gates)

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
        """ Test if inverse function returns the proper set of gates."""

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

    def test_depth(self):
        """ Test depth method on a few circuits """

        c1 = Circuit([Gate("H", 0)]*3 + [Gate("X", 1)])
        self.assertTrue(c1.depth() == 3)

        c2 = Circuit([Gate("H", 0), Gate("CNOT", 1, 0), Gate("CNOT", 2, 1), Gate("H", 0), Gate("CNOT", 0, 2)])
        self.assertTrue(c2.depth() == 4)

        c3 = Circuit()
        self.assertTrue(c3.depth() == 0)

    def test_simple_optimization_functions(self):
        """ Test if removing small rotations and redundant gates return the
        proper set of gates.
        """

        test_circuit = Circuit([Gate("RX", 0, parameter=2.), Gate("CNOT", 1, control=0),
                    Gate("RZ", 2, parameter=0.01), Gate("CNOT", 1, control=0),
                    Gate("RX", 0, parameter=-2.)])
        test_circuit.remove_small_rotations(param_threshold=0.05)

        ref_gates = [Gate("RX", 0, parameter=2.), Gate("CNOT", 1, control=0),
                     Gate("CNOT", 1, control=0), Gate("RX", 0, parameter=-2.)]

        self.assertTrue(ref_gates == test_circuit._gates)

        test_circuit.remove_redundant_gates()

        self.assertTrue([] == test_circuit._gates)

    def test_simple_optimization_minus_a_qubit(self):
        """ Test if removing redundant gates deletes a qubit."""

        test_circuit = Circuit([Gate("X", 0), Gate("H", 1), Gate("H", 1)])
        test_circuit.remove_redundant_gates()

        self.assertEqual(test_circuit.width, 1)


if __name__ == "__main__":
    unittest.main()
