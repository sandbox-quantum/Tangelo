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

from tangelo.toolboxes.ansatz_generator.rucc import RUCC


class RUCCTest(unittest.TestCase):

    def test_construction_rucc(self):
        """Verify behavior of UCC1 and UCC3 construction. Those ansatze are
        constant (they do not change with the system because they always
        represent 4 spin-orbitals).
        """

        ucc1_ansatz = RUCC(1)
        ucc1_ansatz.build_circuit()
        assert(ucc1_ansatz.circuit.counts == {"X": 2, "RX": 2, "H": 6, "CNOT": 6, "RZ": 1})

        ucc3_ansatz = RUCC(3)
        ucc3_ansatz.build_circuit()
        assert(ucc3_ansatz.circuit.counts == {"X": 2, "RX": 4, "H": 6, "CNOT": 8, "RZ": 3})

    def test_ucc1_set_var_params(self):
        """Verify behavior of UCC1 set_var_params for different inputs (keyword,
        list, numpy array).
        """

        ucc1_ansatz = RUCC(1)

        ucc1_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(ucc1_ansatz.var_params, np.array([0.]), decimal=6)

        ucc1_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(ucc1_ansatz.var_params, np.array([1.]), decimal=6)

        ucc1_ansatz.set_var_params([1.])
        np.testing.assert_array_almost_equal(ucc1_ansatz.var_params, np.array([1.]), decimal=6)

        ucc1_ansatz.set_var_params(np.array([1.]))
        np.testing.assert_array_almost_equal(ucc1_ansatz.var_params, np.array([1.]), decimal=6)

    def test_ucc3_set_var_params(self):
        """Verify behavior of UCC3 set_var_params for different inputs (keyword,
        list, numpy array).
        """

        ucc3_ansatz = RUCC(3)

        ucc3_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(ucc3_ansatz.var_params, np.array([0., 0., 0.]), decimal=6)

        ucc3_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(ucc3_ansatz.var_params, np.array([1., 1., 1.]), decimal=6)

        ucc3_ansatz.set_var_params([1., 1., 1.])
        np.testing.assert_array_almost_equal(ucc3_ansatz.var_params, np.array([1., 1., 1.]), decimal=6)

        ucc3_ansatz.set_var_params(np.array([1., 1., 1.]))
        np.testing.assert_array_almost_equal(ucc3_ansatz.var_params, np.array([1., 1., 1.]), decimal=6)

    def test_rucc_wrong_n_params(self):
        """Verify RUCC wrong number of parameters."""

        with self.assertRaises(ValueError):
            RUCC(n_var_params=999)

        with self.assertRaises(ValueError):
            RUCC(n_var_params="3")

        with self.assertRaises(ValueError):
            RUCC(n_var_params=3.141516)

        with self.assertRaises(AssertionError):
            ucc3 = RUCC(n_var_params=3)
            ucc3.build_circuit()
            ucc3.update_var_params([3.1415])


if __name__ == "__main__":
    unittest.main()
