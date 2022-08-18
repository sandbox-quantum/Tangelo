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
import numpy as np
from openfermion.utils import load_operator
from openfermion.linalg import eigenspectrum

from tangelo.helpers.utils import installed_backends
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.operators import translate_operator

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

operator = []
tangelo_op = QubitOperator("X0 Y1 Z2", 1.)


class TranslateOperatorTest(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
