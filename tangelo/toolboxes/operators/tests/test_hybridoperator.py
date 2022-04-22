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

from tangelo.toolboxes.operators import QubitOperator, HybridOperator
from tangelo.toolboxes.operators.hybridoperator import do_commute

qu_op_xyz = QubitOperator("X0 Y1 Z2", 1.)
qu_op_zyz = QubitOperator("Z0 Y1 Z2", 1.)

int_op_xyz = np.array([[2, 3, 1]])
bin_op_xyz = np.array([[1, 1, 0, 0, 1, 1]])


class HybridOperatorUtilitiesTest(unittest.TestCase):

    def test_do_commute(self):
        """Test is_commuting function."""

        commute_xyz_xyz = do_commute(HybridOperator.from_qubitop(qu_op_xyz),
                                       HybridOperator.from_qubitop(qu_op_xyz))
        self.assertTrue(commute_xyz_xyz)

        commute_xyz_zyz = do_commute(HybridOperator.from_qubitop(qu_op_xyz),
                                       HybridOperator.from_qubitop(qu_op_zyz))
        self.assertFalse(commute_xyz_zyz)


class HybridOperatorTest(unittest.TestCase):

    def test_instantiate_from_QubitOperator(self):
        """Test initialization of HybridOperator class with a QubitOperator."""

        hydrib_op = HybridOperator.from_qubitop(qu_op_xyz)

        self.assertDictEqual(hydrib_op.terms, qu_op_xyz.terms)
        np.testing.assert_allclose(hydrib_op.integer, int_op_xyz)
        np.testing.assert_allclose(hydrib_op.binary, bin_op_xyz)

    def test_instantiate_from_integers(self):
        """Test initialization of HybridOperator class with integers."""

        hydrib_op = HybridOperator.from_integerop(int_op_xyz, [1.])

        self.assertDictEqual(hydrib_op.terms, qu_op_xyz.terms)
        np.testing.assert_allclose(hydrib_op.integer, int_op_xyz)
        np.testing.assert_allclose(hydrib_op.binary, bin_op_xyz)

    def test_instantiate_from_binary(self):
        """Test initialization of HybridOperator class with binary numbers."""

        hydrib_op = HybridOperator.from_binaryop(bin_op_xyz, [1.])

        self.assertDictEqual(hydrib_op.terms, qu_op_xyz.terms)
        np.testing.assert_allclose(hydrib_op.integer, int_op_xyz)
        np.testing.assert_allclose(hydrib_op.binary, bin_op_xyz)

    def test_multiply(self):
        """Test multiplication of 2 HybridOperators."""

        hydrib_op_a = HybridOperator.from_qubitop(qu_op_xyz)
        hydrib_op_b = HybridOperator.from_qubitop(qu_op_zyz)

        aa = hydrib_op_a * hydrib_op_a
        np.testing.assert_allclose(aa.integer, np.array([[0, 0, 0]]))
        np.testing.assert_allclose(aa.factors, np.array([1.]))

        ab = hydrib_op_a * hydrib_op_b
        np.testing.assert_allclose(ab.integer, np.array([[3, 0, 0]]))
        np.testing.assert_allclose(ab.factors, np.array([-1.j]))

        ba = hydrib_op_b * hydrib_op_a
        np.testing.assert_allclose(ba.integer, np.array([[3, 0, 0]]))
        np.testing.assert_allclose(ba.factors, np.array([1.j]))


if __name__ == "__main__":
    unittest.main()
