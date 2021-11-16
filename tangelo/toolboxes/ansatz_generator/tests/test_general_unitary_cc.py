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
from scipy.special import binom

from tangelo.toolboxes.ansatz_generator._general_unitary_cc import *


class UCCGSDTest(unittest.TestCase):

    def test_spin_order(self):
        """Test that spin-ordering is implemented correctly, for both
        openfermion and qiskit orderings.
        """
        n_orbitals = 6
        p, q, r, s = 0, 1, 2, 3
        qiskit = np.array([0, 1, 2, 3]), 6 + np.array([0, 1, 2, 3])
        fermion = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
        up_q, dn_q = get_spin_ordered(n_orbitals, p, q, r, s, up_down=True)
        up_f, dn_f = get_spin_ordered(n_orbitals, p, q, r, s, up_down=False)

        self.assertEqual(np.linalg.norm(up_q - qiskit[0]), 0.0, msg="Spin Up Qiskit-Ordering Fails")
        self.assertEqual(np.linalg.norm(dn_q - qiskit[1]), 0.0, msg="Spin Down Qiskit-Ordering Fails")
        self.assertEqual(np.linalg.norm(up_f - fermion[0]), 0.0, msg="Spin Up openfermion-Ordering Fails")
        self.assertEqual(np.linalg.norm(dn_f - fermion[1]), 0.0, msg="Spin Down openfermion-Ordering Fails")

    def count_doubles_groups(self, n_orbs, up_down=False):
        """General test for number of doubles groups found by generator."""
        selection = np.linspace(0, n_orbs - 1, n_orbs, dtype=int)
        groups = np.zeros(5)
        for pp, qq, rr, ss in itertools.product(selection, repeat=4):

            if (pp < qq and pp < rr and pp < ss) and (rr < ss) and (qq != rr and qq != ss):
                _ = get_group_1_2(n_orbs, pp, qq, rr, ss, up_down=up_down)
                groups[0] += 1
            elif qq == rr and pp < ss and pp != qq and ss != qq:
                _ = get_group_1_2(n_orbs, pp, qq, rr, ss, up_down=up_down)
                groups[1] += 1
            elif (pp == qq and qq != rr and rr != ss and ss != pp) and rr < ss:
                _ = get_group_3_4(n_orbs, pp, qq, rr, ss, up_down=up_down)
                groups[2] += 1
            elif pp == qq and qq == rr and rr != ss:
                _ = get_group_3_4(n_orbs, pp, qq, rr, ss, up_down=up_down)
                groups[3] += 1
            elif pp == qq and qq != rr and rr == ss and pp < ss:
                _ = get_group_5(n_orbs, pp, qq, rr, ss, up_down=up_down)
                groups[4] += 1

        self.assertEqual(groups[0], binom(n_orbs, 2) * binom(n_orbs - 2, 2) // 2,
                         msg="{:d} orbs: Invalid Group 1 Number".format(n_orbs))
        self.assertEqual(groups[1], n_orbs * binom(n_orbs - 1, 2),
                         msg="{:d} orbs: Invalid Group 2 Number".format(n_orbs))
        self.assertEqual(groups[2], n_orbs * binom(n_orbs - 1, 2),
                         msg="{:d} orbs: Invalid Group 3 Number".format(n_orbs))
        self.assertEqual(groups[3], 2 * binom(n_orbs, 2), msg="{:d} orbs: Invalid Group 4 Number".format(n_orbs))
        self.assertEqual(groups[4], binom(n_orbs, 2), msg="{:d} orbs: Invalid Group 5 Number".format(n_orbs))
        self.assertEqual(np.sum(groups), get_doubles_number(n_orbs),
                         msg="{:d} orbs: Invalid Total Number".format(n_orbs))

    def test_count_doubles(self):
        """Test for checking number of doubles excitations generated."""
        self.count_doubles_groups(5, up_down=False)
        self.count_doubles_groups(5, up_down=True)
        self.count_doubles_groups(8, up_down=False)

    def test_count_singles(self):
        """Test for checking number of singles excitations generated."""
        self.assertEqual(len(get_singles(5)), get_singles_number(5), msg="5 orbs: Invalid Number of singles")
        self.assertEqual(len(get_singles(10)), get_singles_number(10), msg="5 orbs: Invalid Number of singles")

    def test_coeff_pass(self):
        """Test that errors are handled correctly when an invalid number of
        coefficients are passed. Also tests that good input returns good output.
        """
        n_qubits = 4

        # Get expected number of single- and double-coefficients
        n_single_coeffs = get_singles_number(n_qubits // 2)
        n_double_coeffs = get_doubles_number(n_qubits // 2)

        # Generate good coefficients
        good_single_coeffs = np.random.random(n_single_coeffs)
        good_double_coeffs = np.random.random(n_double_coeffs)
        # Generate bad coefficients
        bad_single_coeffs = np.random.random(n_single_coeffs + 10)
        bad_double_coeffs = np.random.random(n_double_coeffs + 10)

        # Combine singles and doubles
        good_coeffs = np.concatenate((good_single_coeffs, good_double_coeffs))

        # Check that an invalid input raises a ValueError in each possible case
        with self.assertRaises(ValueError):
            get_coeffs(n_qubits, bad_single_coeffs, good_double_coeffs)
        with self.assertRaises(ValueError):
            get_coeffs(n_qubits, bad_single_coeffs, good_double_coeffs)
        with self.assertRaises(ValueError):
            get_coeffs(n_qubits, bad_single_coeffs, bad_double_coeffs)

        # Check that a valid input gives the expected result
        good_result = get_coeffs(n_qubits, good_single_coeffs, good_double_coeffs)
        self.assertTrue(np.allclose(good_coeffs, good_result))


if __name__ == "__main__":
    unittest.main()
