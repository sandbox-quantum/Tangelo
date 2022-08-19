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

tangelo_op = QubitOperator("X0 Y1 Z2", 1.)


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

        from qiskit.opflow.primitive_ops import PauliSumOp
        qiskit_op = PauliSumOp.from_list([("ZYX", 1.)])

        test_op = translate_operator(qiskit_op, source="qiskit", target="tangelo")
        self.assertEqual(test_op, tangelo_op)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_tangelo_to_qiskit(self):
        """Test translation from a tangelo to a qiskit operator."""

        from qiskit.opflow.primitive_ops import PauliSumOp
        qiskit_op = PauliSumOp.from_list([("ZYX", 1.)])

        test_op = translate_operator(tangelo_op, source="tangelo", target="qiskit")
        self.assertEqual(qiskit_op, test_op)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_tangelo_to_qiskit_H2_eigenvalues(self):
        """Test eigenvalues resulting from a tangelo to qiskit translation."""

        from qiskit.algorithms import NumPyEigensolver

        qu_op = load_operator("H2_JW_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        test_op = translate_operator(qu_op, source="tangelo", target="qiskit")

        eigenvalues_tangelo = eigenspectrum(qu_op)

        qiskit_solver = NumPyEigensolver(2**4)
        eigenvalues_qiskit = qiskit_solver.compute_eigenvalues(test_op)

        np.testing.assert_array_almost_equal(eigenvalues_tangelo, eigenvalues_qiskit.eigenvalues)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Qiskit not available \n")
    def test_tangelo_to_qiskit_big(self):
        """Test translation from a tangelo to a qiskit operator, for a large input"""

        n_qubits = 10
        n_terms = 3**n_qubits

        # Build large operator made of all possible "full" Pauli words (no 'I') of length n_qubits
        terms = {tuple(zip(range(n_qubits), pw)): 1.0 for pw in product(['X', 'Y', 'Z'], repeat=n_qubits)}
        q_op = QubitOperator()
        q_op.terms = terms

        s, t = "tangelo", "qiskit"
        tstart1 = time.time()
        tmp_op = translate_operator(q_op, source=s, target=t)
        tstop1 = time.time()
        print(f"Qubit operator conversion {s} to {t}: {tstop1 - tstart1:.1f} (terms = {n_terms})")

        t, s = s, t
        tstart2 = time.time()
        q_op2 = translate_operator(tmp_op, source=s, target=t)
        tstop2 = time.time()
        print(f"Qubit operator conversion {s} to {t}: {tstop2 - tstart2:.1f} (terms = {n_terms})")

        assert(q_op == q_op2)


if __name__ == "__main__":
    unittest.main()
