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

import unittest

from openfermion.transforms import bravyi_kitaev as openfermion_bravyi_kitaev
from tangelo.toolboxes.qubit_mappings.bravyi_kitaev import bravyi_kitaev
from tangelo.toolboxes.operators import FermionOperator, QubitOperator


class BravyiKitaevTest(unittest.TestCase):

    def test_few_qubits(self):
        """Test that an error is raised if the number of qubits specified for an
        operator is too few.
        """
        # Instantiate simple non-trivial FermionOperator input
        input_operator = FermionOperator(((0, 0), (1, 0), (5, 0)))
        n_qubits = 3
        with self.assertRaises(ValueError):
            bravyi_kitaev(input_operator, n_qubits)

    def test_input_raise(self):
        """Test that invalid operator type throws an error."""
        input_operator = QubitOperator((1, 'X'))
        with self.assertRaises(TypeError):
            bravyi_kitaev(input_operator, n_qubits=2)

    def test_openfermion_equivalence(self):
        """Test that our wrapper returns the same result as openfermion's bare
        implementation of bravyi_kitaev.
        """
        # Instantiate simple non-trivial FermionOperator input
        input_operator = FermionOperator(((0, 0), (1, 0), (2, 0), (12, 1)))
        input_operator += FermionOperator((13, 1), 0.2)
        n_qubits = 14

        tangelo_result = bravyi_kitaev(input_operator, n_qubits=n_qubits)
        openfermion_result = openfermion_bravyi_kitaev(input_operator, n_qubits=n_qubits)

        # check that the number of terms is the same.
        self.assertEqual(len(tangelo_result.terms), len(openfermion_result.terms), msg="Number of terms generated does not agree"
                                                                                       "with openfermion implementation of Bravyi Kitaev.")

        # check that the term coefficients are the same
        for ti in tangelo_result.terms:
            factor = tangelo_result.terms[ti]
            openfermion_factor = openfermion_result.terms[ti]
            self.assertEqual(factor, openfermion_factor, msg="Term coefficient does not agree with openfermion bravyi_kitaev.")


if __name__ == "__main__":
    unittest.main()
