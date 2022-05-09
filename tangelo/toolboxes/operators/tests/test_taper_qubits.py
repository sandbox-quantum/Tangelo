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
from openfermion.linalg import qubit_operator_sparse
from openfermion.utils import load_operator

from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.operators.taper_qubits import QubitTapering

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class QubitTaperingTest(unittest.TestCase):

    def test_unsupported_mapping(self):
        """Test unsupported mapping for qubit for tapering."""

        with self.assertRaises(NotImplementedError):
            QubitTapering(QubitOperator(), 4, 2, 0, "scBK", True)

    def test_z2taper_h2_jw_occupied_first(self):
        """Test Z2 tapering of H2 JW up_then_down=False."""

        qu_op = load_operator("H2_JW_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "JW", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_jw_spinup_first(self):
        """Test Z2 tapering of H2 JW up_then_down=True."""

        qu_op = load_operator("H2_JW_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "JW", True)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_bk_occupied_first(self):
        """Test Z2 tapering of H2 BK up_then_down=False."""

        qu_op = load_operator("H2_BK_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "BK", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_bk_spinup_first(self):
        """Test Z2 tapering of H2 BK up_then_down=True."""

        qu_op = load_operator("H2_BK_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "BK", True)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_jkmn_occupied_first(self):
        """Test Z2 tapering of H2 JKMN up_then_down=False."""

        qu_op = load_operator("H2_JKMN_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "JKMN", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_jkmn_spinup_first(self):
        """Test Z2 tapering of H2 JKMN up_then_down=True."""

        qu_op = load_operator("H2_JKMN_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)
        e = np.min(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))

        tapering = QubitTapering(qu_op, 4, 2, 0, "JKMN", True)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))

        self.assertAlmostEqual(e, e_taper, places=5)

    def test_z2taper_h2_triplet(self):
        """Test Z2 tapering of H2 triplet JW up_then_down=False."""

        qu_op = load_operator("H2_JW_occfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)

        # Triplet state is the second lowest energy after the groud state.
        e = np.sort(np.linalg.eigvalsh(qubit_operator_sparse(qu_op).todense()))[1]

        tapering = QubitTapering(qu_op, 4, 2, 2, "JW", False)
        tapered_qu_op = tapering.z2_tapered_op.qubitoperator
        e_taper = np.min(np.linalg.eigvalsh(qubit_operator_sparse(tapered_qu_op).todense()))
        self.assertAlmostEqual(e, e_taper, places=5)


if __name__ == "__main__":
    unittest.main()
