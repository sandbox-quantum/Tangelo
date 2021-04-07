import unittest
import numpy as np

from qsdk.toolboxes.operators import QubitOperator,FermionOperator

from qsdk.toolboxes.qubit_mappings import bravyi_kitaev,jordan_wigner
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


class MappingTest(unittest.TestCase):

    def test_arg_types(self):
        """Test that different mapping specifiers return a valid, identical result."""
        fermion = FermionOperator(((1,0),(2,1)),1.0) + FermionOperator(((0,1),(3,0)),0.5)
        n_qubits = 4
        jw_input = ['Jordan_Wigner','JW','jw',0]
        bk_input = ['Bravyi_Kitaev','BK','bk',1]
        jw_string = fermion_to_qubit_mapping(fermion, mapping = jw_input[0])
        for ii in range(1,len(jw_input)):
            jw_test = fermion_to_qubit_mapping(fermion, mapping = jw_input[ii])
            self.assertEquals(jw_string,jw_test)
        bk_string = fermion_to_qubit_mapping(fermion, mapping = bk_input[0], n_qubits = n_qubits)
        for ii in range(1,len(jw_input)):
            bk_test = fermion_to_qubit_mapping(fermion, mapping = bk_input[ii], n_qubits = n_qubits)
            self.assertEquals(bk_string,bk_test)

    def test_handle_invalid_mapping(self):

        fermion = FermionOperator(((1,0),(2,1)),1.0) + FermionOperator(((0,1),(3,0)),0.5)
        with self.assertRaises(ValueError):
            garbage_mapping = fermion_to_qubit_mapping(fermion, mapping = "bogus")

        



if __name__ == "__main__":

    unittest.main()