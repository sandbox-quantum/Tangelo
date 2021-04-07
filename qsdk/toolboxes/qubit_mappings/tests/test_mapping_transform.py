"""Simple tests for input to mapping_transform.py.
Presently, mostly using wrappers on openfermion code, so pretty minimal testing here.
"""
import unittest
import numpy as np

from qsdk.toolboxes.operators import QubitOperator,FermionOperator
from qsdk.toolboxes.qubit_mappings import bravyi_kitaev,jordan_wigner
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


class MappingTest(unittest.TestCase):

    def test_bk(self):
        """Check output from Bravyi-Kitaev transformation"""
        bk_operator = QubitOperator(((0, 'Z'), (1, 'Y'), (2, 'X')), -0.25j)
        bk_operator += QubitOperator(((0, 'Z'), (1, 'Y'), (2, 'Y')), -0.25)
        bk_operator += QubitOperator(((1, 'X'), (2, 'X')), -0.25)
        bk_operator += QubitOperator((((1, 'X'), (2, 'Y'))), 0.25j)
        bk_operator += QubitOperator(((0, 'X'), (1, 'Y'), (2, 'Z')), -0.125j)
        bk_operator += QubitOperator(((0, 'X'), (1, 'X'), (3, 'Z')), -0.125)
        bk_operator += QubitOperator(((0, 'Y'), (1, 'Y'), (2, 'Z')), -0.125) 
        bk_operator += QubitOperator(((0, 'Y'), (1, 'X'), (3, 'Z')), 0.125j)

        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        n_qubits = 4
        qubit = fermion_to_qubit_mapping(fermion, mapping = 'BK', n_qubits = n_qubits)
        self.assertEquals(qubit,bk_operator)

    def test_jw(self):
        """Check output from Bravyi-Kitaev transformation"""
        jw_operator = QubitOperator(((1, 'Y'), (2, 'X')), -0.25j)
        jw_operator += QubitOperator(((1, 'Y'), (2, 'Y')), -0.25)
        jw_operator += QubitOperator(((1, 'X'), (2, 'X')), -0.25)
        jw_operator += QubitOperator(((1, 'X'), (2, 'Y')), 0.25j)
        jw_operator += QubitOperator(((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'X')), -0.125j)
        jw_operator += QubitOperator(((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Y')), 0.125)
        jw_operator += QubitOperator(((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'X')), 0.125)
        jw_operator += QubitOperator(((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y')), 0.125j)
        
        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        qubit = fermion_to_qubit_mapping(fermion, mapping = 'JW')
        self.assertEquals(qubit,jw_operator)


    def test_handle_invalid_mapping(self):
        """Test that error is handled if invalid mapping is requested."""
        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        with self.assertRaises(ValueError):
            fermion_to_qubit_mapping(fermion, mapping = "bogus")


if __name__ == "__main__":
    unittest.main()
