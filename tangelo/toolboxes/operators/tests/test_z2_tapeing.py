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
from openfermion.linalg import qubit_operator_sparse
from openfermion.utils import load_operator

from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.operators.taper_qubits import QubitTapering


class Z2TaperingTest(unittest.TestCase):

    def test_unsupported_mapping(self):
        """Test unsupported mapping for qubit z2 tapering."""

        with self.assertRaises(NotImplementedError):
            QubitTapering(QubitOperator(), 4, 2, "scBK", True)

    def test_taper_h2_jw_occupied_first(self):
        """Test tapering of H2 JW up_then_down=False."""

        qu_op = load_operator("H2_JW_occfirst.data", data_directory="data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, "JW", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_taper_h2_jw_spinup_first(self):
        """Test tapering of H2 JW up_then_down=True."""

        qu_op = load_operator("H2_JW_spinupfirst.data", data_directory="data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, "JW", True)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_taper_h2_bk_occupied_first(self):
        """Test tapering of H2 BK up_then_down=False."""

        qu_op = load_operator("H2_BK_occfirst.data", data_directory="data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, "BK", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_taper_h2_bk_spinup_first(self):
        """Test tapering of H2 BK up_then_down=True."""

        qu_op = load_operator("H2_BK_spinupfirst.data", data_directory="data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, "BK", True)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)


if __name__ == "__main__":
    unittest.main()
