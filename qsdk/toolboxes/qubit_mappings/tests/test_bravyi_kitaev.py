import unittest
import numpy as np
from scipy.special import binom
import itertools

from openfermion.transforms import bravyi_kitaev as openfermion_bravyi_kitaev
from openfermion.utils import count_qubits
from qsdk.toolboxes.qubit_mappings.bravyi_kitaev import bravyi_kitaev,count_qubits
from qsdk.toolboxes.operators import FermionOperator,QubitOperator


class BravyiKitaevTest(unittest.TestCase):

    def test_few_qubits(self):
        """Test that an error is raised if the number of qubits specified for an operator is too few."""
        #Instantiate simple non-trivial FermionOperator input
        input_operator = FermionOperator(((0, 0), (1, 0), (5, 0)))
        n_qubits = 3
        with self.assertRaises(ValueError):
            bravyi_kitaev(input_operator, n_qubits)


    def test_input_raise(self):
        """Test that invalid operator type throws an error."""
        input_operator = QubitOperator((1,'X'))
        with self.assertRaises(TypeError):
            bravyi_kitaev(input_operator,n_qubits = 2)


    def test_openfermion_equivalence(self):
        """Test that our wrapper returns the same result as openfermion's bare implementation of bravyi_kitaev."""
        #Instantiate simple non-trivial FermionOperator input
        input_operator = FermionOperator(((0, 0), (1, 0), (2, 0), (12, 1)))
        input_operator += FermionOperator((13, 1), 0.2)
        n_qubits = 14

        qsdk_result = bravyi_kitaev(input_operator, n_qubits = n_qubits)
        openfermion_result = openfermion_bravyi_kitaev(input_operator, n_qubits = n_qubits)
        
        #check that the number of terms is the same.
        self.assertEqual(len(qsdk_result.terms), len(openfermion_result.terms), msg = "Number of terms generated does not agree with openfermion implementation of Bravyi Kitaev.")
        
        #check that the term coefficients are the same
        for ti in qsdk_result.terms:
            factor = qsdk_result.terms[ti]
            openfermion_factor = openfermion_result.terms[ti]
            self.assertEqual(factor, openfermion_factor, msg = "Term coefficient does not agree with openfermion bravyi_kitaev.")


if __name__ == "__main__":

    unittest.main()
