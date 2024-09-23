# Copyright SandboxAQ 2021-2024.
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

from tangelo.molecule_library import mol_H10_321g
from tangelo.toolboxes.circuits.qre import qre_benchq, qre_pennylane, qre_pyliqtr


class QRETest(unittest.TestCase):

    @unittest.skip("Installing benchq enforces an old qiskit version that breaks our qiskit interface.")
    def test_qre_benchq(self):
        """Test the QRE output type for the benchq interface."""

        n_toffolis, n_qubits = qre_benchq(mol_H10_321g, 1e-6)

        self.assertIsInstance(n_toffolis, int)
        self.assertIsInstance(n_qubits, int)

        # As done in tests/benchq/problem_embeddings/test_qpe.py in benchq repo.
        self.assertGreater(n_qubits, mol_H10_321g.n_active_sos)

    def test_qre_pennylane(self):
        """Test the QRE output type for the pennylane interface."""

        output = qre_pennylane(mol_H10_321g)

        self.assertIsInstance(output.gates, int)
        self.assertIsInstance(output.qubits, int)

    @unittest.skip("Installing pyLIQTR requires and old cirq version that breaks linq tests.")
    def test_qre_pyliqtr(self):
        """Test the QRE output type for the pyLIQTR interface."""

        output = qre_pyliqtr(mol_H10_321g)

        self.assertIsInstance(output.get("LogicalQubits"), int)
        self.assertIsInstance(output.get("T"), int)
        self.assertIsInstance(output.get("Clifford"), int)
        self.assertIsInstance(output.get("lambda"), float)


if __name__ == "__main__":
    unittest.main()
