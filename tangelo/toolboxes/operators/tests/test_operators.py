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

from tangelo.toolboxes.operators import QubitHamiltonian, QubitOperator, count_qubits, qubitop_to_qubitham


class OperatorsUtilitiesTest(unittest.TestCase):

    def test_count_qubits(self):
        """Test count_qubits function."""

        n_qubits_3 = count_qubits(QubitHamiltonian("JW", True, term="X0 Y1 Z2"))
        self.assertEqual(n_qubits_3, 3)

        n_qubits_5 = count_qubits(QubitHamiltonian("JW", True, term="X0 Y1 Z4"))
        self.assertEqual(n_qubits_5, 5)

    def test_qubitop_to_qubitham(self):
        """Test qubitop_to_qubitham function."""

        qubit_ham = qubitop_to_qubitham(0.5*QubitOperator("X0 Y1 Z2"), "BK", False)

        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)


class QubitHamiltonianTest(unittest.TestCase):

    def test_instantiate_QubitHamiltonian(self):
        """Test initialization of QubitHamiltonian class."""

        qubit_ham = 0.5 * QubitHamiltonian("BK", False, term="X0 Y1 Z2")
        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)

    def test_instantiate_QubitHamiltonian(self):
        """Test error raising when 2 incompatible QubitHamiltonian are summed up
        together.
        """

        qubit_ham = QubitHamiltonian("JW", True, term="X0 Y1 Z2")

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("BK", True, term="Z0 X1 Y2")

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("JW", False, term="Z0 X1 Y2")

    def test_to_qubit_operator(self):
        """Test exportation of QubitHamiltonian to QubitOperator."""

        qubit_ham = 1. * QubitHamiltonian("JW", True, term="X0 Y1 Z2")

        self.assertEqual(qubit_ham.to_qubitoperator(),
            QubitOperator(term="X0 Y1 Z2", coefficient=1.))


if __name__ == "__main__":
    unittest.main()
