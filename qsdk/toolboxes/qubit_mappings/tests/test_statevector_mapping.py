"""Tests for statevector mapping methods, which carry a numpy array indicating fermionic
occupation of reference state into qubit representation."""
import unittest
import numpy as np

from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_vector,vector_to_circuit
from agnostic_simulator import Circuit,Gate

class TestVector(unittest.TestCase):

    def test_jw_value(self):
        """Check that Jordan-Wigner mapping returns correct vector, for both default spin orderings"""
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_updown = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        output_jw = get_vector(vector.size, sum(vector), mapping = 'jw', updown = False)
        output_jw_updown = get_vector(vector.size, sum(vector), mapping = 'jw', updown = True)
        self.assertEqual(np.linalg.norm(vector - output_jw), 0.0)
        self.assertEqual(np.linalg.norm(vector_updown - output_jw_updown), 0.0)


    def test_bk_value(self):
        """Check that Bravyi-Kitaev mapping returns correct vector, for both default spin orderings"""
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_bk = np.array([1, 0, 1, 0, 0, 0, 0, 0])
        vector_bk_updown = np.array([1, 0, 0, 0, 1, 0, 0, 0])

        output_bk = get_vector(vector.size, sum(vector), mapping = 'bk', updown = False)
        output_bk_updown = get_vector(vector.size, sum(vector), mapping = 'bk', updown = True)
        self.assertEqual(np.linalg.norm(vector_bk - output_bk), 0.0)
        self.assertEqual(np.linalg.norm(vector_bk_updown - output_bk_updown), 0.0)


    def test_circuit_width(self):
        """If fixed-width circuit is passed as input, raise error if circuit incompatible with vector"""
        vector = np.array([1, 1, 1, 1, 0, 0, 1, 1])
        circuit = Circuit(n_qubits = 6)
        with self.assertRaises(ValueError):
            vector_to_circuit(vector, circuit = circuit)


    def test_circuit_build(self):
        """Check circuit width and size (number of X gates)."""
        vector = np.array([1, 1, 1, 1, 0, 0, 1, 1])
        circuit = vector_to_circuit(vector)
        self.assertEqual(circuit.size, sum(vector))
        self.assertEqual(circuit.width, vector.size)
        

if __name__ == "__main__":
    unittest.main()
