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

import os
import unittest
import time
from itertools import product

import numpy as np
from openfermion.utils import load_operator
from openfermion.linalg import eigenspectrum

from tangelo.helpers.utils import installed_backends
from tangelo.linq import translate_operator
from tangelo.toolboxes.operators import QubitOperator

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

tangelo_op = QubitOperator("X0 Y1 Z2", 2.) + QubitOperator("", 3.)


class TranslateOperatorTest(unittest.TestCase):

    def test_unsupported_source(self):
        """Test error with an unsuported source."""

        with self.assertRaises(NotImplementedError):
            translate_operator(tangelo_op, source="sourcenotsupported", target="tangelo")

    def test_unsupported_target(self):
        """Test error with an unsuported target."""

        with self.assertRaises(NotImplementedError):
            translate_operator(tangelo_op, source="tangelo", target="targetnotsupported")

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_qiskit_to_tangelo(self):
        """Test translation from a qiskit to a tangelo operator."""

        from qiskit.quantum_info.operators import SparsePauliOp
        qiskit_op = SparsePauliOp.from_list([("ZYX", 2.), ("III", 3.)])

        test_op = translate_operator(qiskit_op, source="qiskit", target="tangelo")
        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_tangelo_to_qiskit(self):
        """Test translation from a tangelo to a qiskit operator."""

        from qiskit.quantum_info.operators import SparsePauliOp
        qiskit_op = SparsePauliOp.from_list([("ZYX", 2.), ("III", 3.)])

        test_op = translate_operator(tangelo_op, source="tangelo", target="qiskit")
        self.assertEqual(qiskit_op, test_op)

    @unittest.skipIf("cirq" not in installed_backends, "Test Skipped: Cirq not available \n")
    def test_cirq_to_tangelo(self):
        """Test translation from a cirq to a tangelo operator."""

        from cirq import PauliSum, PauliString, LineQubit, X, Y, Z
        qubit_a, qubit_b, qubit_c = LineQubit.range(3)
        cirq_op = PauliSum.from_pauli_strings([
            PauliString(2., X(qubit_a), Y(qubit_b), Z(qubit_c)),
            PauliString(3.)
        ])

        test_op = translate_operator(cirq_op, source="cirq", target="tangelo")
        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("cirq" not in installed_backends, "Test Skipped: Cirq not available \n")
    def test_tangelo_to_cirq(self):
        """Test translation from a tangelo to a cirq operator."""

        from cirq import PauliSum, PauliString, LineQubit, X, Y, Z
        qubit_a, qubit_b, qubit_c = LineQubit.range(3)
        cirq_op = PauliSum.from_pauli_strings([
            PauliString(2., X(qubit_a), Y(qubit_b), Z(qubit_c)),
            PauliString(3.)
        ])

        test_op = translate_operator(tangelo_op, source="tangelo", target="cirq")
        self.assertEqual(cirq_op, test_op)

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: qulacs not available \n")
    def test_qulacs_to_tangelo(self):
        """Test translation from a qulacs to a tangelo operator."""

        from qulacs import GeneralQuantumOperator
        qulacs_op = GeneralQuantumOperator(3)
        qulacs_op.add_operator(2., "X 0 Y 1 Z 2")
        qulacs_op.add_operator(3., "")

        test_op = translate_operator(qulacs_op, source="qulacs", target="tangelo")
        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: qulacs not available \n")
    def test_tangelo_to_qulacs(self):
        """Test translation from a tangelo to a qulacs operator."""

        from qulacs import GeneralQuantumOperator
        qulacs_op = GeneralQuantumOperator(3)
        qulacs_op.add_operator(2., "X 0 Y 1 Z 2")
        qulacs_op.add_operator(3., "")

        test_op = translate_operator(tangelo_op, source="tangelo", target="qulacs")

        n_terms = qulacs_op.get_term_count()
        coeffs = [qulacs_op.get_term(i).get_coef() for i in range(n_terms)]
        terms = [qulacs_op.get_term(i).get_pauli_string() for i in range(n_terms)]

        # __eq__ method is not implemented for GeneralQuantumOperator. It is
        # checking the addresses in memory when comparing 2 qulacs objects.
        self.assertEqual(n_terms, test_op.get_term_count())
        for i in range(test_op.get_term_count()):
            self.assertIn(test_op.get_term(i).get_coef(), coeffs)
            self.assertIn(test_op.get_term(i).get_pauli_string(), terms)

    @unittest.skipIf("pennylane" not in installed_backends, "Test Skipped: Pennylane not available \n")
    def test_pennylane_to_tangelo(self):
        """Test translation from a pennylane to a tangelo operator."""

        import pennylane as qml
        coeffs = [2., 3.]
        obs = [
            qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2),
            qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2)
        ]
        pennylane_H = qml.Hamiltonian(coeffs, obs)

        test_op = translate_operator(pennylane_H, source="pennylane", target="tangelo")
        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("pennylane" not in installed_backends, "Test Skipped: Pennylane not available \n")
    def test_tangelo_to_pennylane(self):
        """Test translation from a tangelo to a pennylane operator."""

        import pennylane as qml
        coeffs = [2., 3.]
        obs = [
            qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2),
            qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2)
        ]
        pennylane_H = qml.Hamiltonian(coeffs, obs)

        test_op = translate_operator(tangelo_op, source="tangelo", target="pennylane")

        # __eq__ method not implemented in pennylane.Hamiltonian.
        self.assertTrue(pennylane_H.compare(test_op))

    @unittest.skipIf("projectq" not in installed_backends, "Test Skipped: ProjectQ not available \n")
    def test_projectq_to_tangelo(self):
        """Test translation from a projectq to a tangelo operator."""

        from projectq.ops import QubitOperator as ProjectQQubitOperator
        projectq_op = ProjectQQubitOperator("X0 Y1 Z2", 2.) + ProjectQQubitOperator("", 3.)

        test_op = translate_operator(projectq_op, source="projectq", target="tangelo")

        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("projectq" not in installed_backends, "Test Skipped: ProjectQ not available \n")
    def test_tangelo_to_projectq(self):
        """Test translation from a tangelo to a projectq operator."""

        from projectq.ops import QubitOperator as ProjectQQubitOperator
        projectq_op = ProjectQQubitOperator("X0 Y1 Z2", 2.) + ProjectQQubitOperator("", 3.)

        test_op = translate_operator(tangelo_op, source="tangelo", target="projectq")

        self.assertEqual(projectq_op, test_op)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_tangelo_to_qiskit_H2_eigenvalues(self):
        """Test eigenvalues resulting from a tangelo to qiskit translation."""

        qu_op = load_operator("H2_JW_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        test_op = translate_operator(qu_op, source="tangelo", target="qiskit")

        eigenvalues_tangelo = eigenspectrum(qu_op)

        eigenvalues_qiskit = np.linalg.eigh(test_op.to_matrix(sparse=False))[0]

        np.testing.assert_array_almost_equal(eigenvalues_tangelo, eigenvalues_qiskit)


if __name__ == "__main__":
    unittest.main()
