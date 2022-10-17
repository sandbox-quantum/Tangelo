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

import numpy as np

from tangelo.toolboxes.operators import QubitHamiltonian, FermionOperator, QubitOperator, count_qubits, qubitop_to_qubitham


class OperatorsUtilitiesTest(unittest.TestCase):

    def test_count_qubits(self):
        """Test count_qubits function."""

        n_qubits_3 = count_qubits(QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True))
        self.assertEqual(n_qubits_3, 3)

        n_qubits_5 = count_qubits(QubitHamiltonian("X0 Y1 Z4", mapping="JW", up_then_down=True))
        self.assertEqual(n_qubits_5, 5)

    def test_qubitop_to_qubitham(self):
        """Test qubitop_to_qubitham function."""

        qubit_ham = qubitop_to_qubitham(0.5*QubitOperator("X0 Y1 Z2"), "BK", False)

        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)


class FermionOperatorTest(unittest.TestCase):

    def test_get_coeff(self):
        """Test the get_coeff method of the FermionOperator class."""

        # Constant.
        dummy_ferm_op = FermionOperator("", 1.)

        # 1-body terms.
        dummy_ferm_op += FermionOperator("0^ 0", 2.)
        dummy_ferm_op += FermionOperator("0^ 1", 3.)
        dummy_ferm_op += FermionOperator("1^ 0", 4.)
        dummy_ferm_op += FermionOperator("1^ 1", 5.)

        # 2-body term.
        dummy_ferm_op += FermionOperator("0^ 0^ 0 0", 7.)
        dummy_ferm_op += FermionOperator("1^ 1^ 1 1", 8.)
        dummy_ferm_op += FermionOperator("0^ 1^ 0 1", 9.)

        ref_cte = 1.
        ref_one_body = np.arange(2, 6, 1.).reshape((2, 2))
        ref_two_body = np.zeros((2, 2, 2, 2))
        ref_two_body[0, 0, 0, 0] = 7.
        ref_two_body[1, 1, 1, 1] = 8.
        ref_two_body[0, 1, 0, 1] = 9.

        cte, one_body, two_body = dummy_ferm_op.get_coeffs()

        self.assertAlmostEqual(ref_cte, cte)
        np.testing.assert_array_almost_equal(ref_one_body, one_body)
        np.testing.assert_array_almost_equal(ref_two_body, two_body)


class QubitHamiltonianTest(unittest.TestCase):

    def test_instantiate_QubitHamiltonian(self):
        """Test initialization of QubitHamiltonian class."""

        qubit_ham = 0.5 * QubitHamiltonian("X0 Y1 Z2", mapping="BK", up_then_down=False)
        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)

    def test_instantiate_QubitHamiltonian(self):
        """Test error raising when 2 incompatible QubitHamiltonian are summed up
        together.
        """

        qubit_ham = QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True)

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("Z0 X1 Y2", mapping="BK", up_then_down=True)

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("Z0 X1 Y2", mapping="JW",  up_then_down=False)

    def test_to_qubit_operator(self):
        """Test exportation of QubitHamiltonian to QubitOperator."""

        qubit_ham = 1. * QubitHamiltonian("X0 Y1 Z2", mapping="JW",  up_then_down=True)

        self.assertEqual(qubit_ham.to_qubitoperator(),
            QubitOperator(term="X0 Y1 Z2", coefficient=1.))

    def test_get_operators(self):
        """Test get_operators methods, defined in QubitOperator class."""

        terms = [QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True),
                 QubitHamiltonian("Z0 X1 Y2", mapping="JW", up_then_down=True),
                 QubitHamiltonian("Y0 Z1 X2", mapping="JW", up_then_down=True)]

        H = terms[0] + terms[1] + terms[2]

        for ref_term, term in zip(terms, H.get_operators()):
            self.assertEqual(ref_term, term)


if __name__ == "__main__":
    unittest.main()
