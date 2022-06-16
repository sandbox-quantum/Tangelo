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

from tangelo.toolboxes.operators import MultiformOperator
from tangelo.toolboxes.operators.z2_tapering import get_clifford_operators, get_unitary, get_eigenvalues


class Z2TaperingHelperFunctionsTest(unittest.TestCase):

    def test_get_clifford_operators(self):
        """Test get_clifford_operators function with a given kernel."""

        kernel = np.array([
            [0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0]
        ])
        clifford_ops, indices = get_clifford_operators(kernel)

        self.assertEqual(3, len(clifford_ops))

        np.testing.assert_array_equal(np.array([3, 1, 2]), indices)

        ref_clif1 = {((3, 'X'),): np.sqrt(0.5), ((0, 'Z'), (3, 'Z')): np.sqrt(0.5)}
        self.assertDictEqual(ref_clif1, clifford_ops[0].terms)

        ref_clif2 = {((1, 'X'),): np.sqrt(0.5), ((0, 'Z'), (1, 'Z')): np.sqrt(0.5)}
        self.assertDictEqual(ref_clif2, clifford_ops[1].terms)

        ref_clif3 = {((2, 'X'),): np.sqrt(0.5), ((0, 'Z'), (2, 'Z')): np.sqrt(0.5)}
        self.assertDictEqual(ref_clif3, clifford_ops[2].terms)

    def test_get_unitary(self):
        """Test get_unitary function with a given set of Clifford operators."""

        ref_u_terms = {
            ((1, 'X'), (2, 'X'), (3, 'X')): (0.35355339059327384+0j),
            ((0, 'Z'), (1, 'X'), (2, 'Z'), (3, 'X')): (0.35355339059327384+0j),
            ((0, 'Z'), (1, 'Z'), (2, 'X'), (3, 'X')): (0.35355339059327384+0j),
            ((1, 'Z'), (2, 'Z'), (3, 'X')): (0.35355339059327384+0j),
            ((0, 'Z'), (1, 'X'), (2, 'X'), (3, 'Z')): (0.35355339059327384+0j),
            ((1, 'X'), (2, 'Z'), (3, 'Z')): (0.35355339059327384+0j),
            ((1, 'Z'), (2, 'X'), (3, 'Z')): (0.35355339059327384+0j),
            ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z')): (0.35355339059327384+0j)
        }

        factors = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        clif1 = np.array([
            [1, 0, 0, 1],
            [0, 0, 0, 2]
        ])
        clif2 = np.array([
            [1, 1, 0, 0],
            [0, 2, 0, 0]
        ])
        clif3 = np.array([
            [1, 0, 1, 0],
            [0, 0, 2, 0]
        ])

        cliff_ops = [
            MultiformOperator.from_integerop(clif1, factors),
            MultiformOperator.from_integerop(clif2, factors),
            MultiformOperator.from_integerop(clif3, factors),
        ]

        U = get_unitary(cliff_ops)
        self.assertDictEqual(ref_u_terms, U.terms)

    def test_get_eigenvalues(self):
        """Test get_eigenvalues function with a given kernel operator
        representing symmetries.
        """

        integers = np.array([
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ])
        kernel_op = MultiformOperator.from_integerop(integers, np.ones(integers.shape[0]))

        np.testing.assert_array_equal([-1, 1, -1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "JW", False))
        np.testing.assert_array_equal([-1, -1, 1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "JW", True))
        np.testing.assert_array_equal([-1, -1, -1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "BK", False))
        np.testing.assert_array_equal([-1, 1, 1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "BK", True))
        np.testing.assert_array_equal([1, -1, 1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "JKMN", False))
        np.testing.assert_array_equal([1, 1, -1], get_eigenvalues(kernel_op.binary, 4, 2, 0, "JKMN", True))


if __name__ == "__main__":
    unittest.main()
