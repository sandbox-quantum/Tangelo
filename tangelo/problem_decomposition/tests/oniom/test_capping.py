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

from tangelo.problem_decomposition.oniom._helpers.helper_classes import Link
from tangelo.molecule_library import xyz_ethane
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list


class ONIOMCappingTest(unittest.TestCase):

    def test_unsupported_string_species(self):
        """Test unsupported built-in chemical group (raise ValueError)."""

        with self.assertRaises(ValueError):
            Link(0, 4, factor=1., species="UNSUPPORTED")

    def test_unsupported_custom_species(self):
        """Test unsupported custom chemical group (raise ValueError)."""

        # No ghost atom (X) as first element.
        not_supported_chem_group = [
            ["C", [ 0.71660,  0.89800,  0.64250]],
            ["I", [ 0.53970,  1.76660, -0.00250]],
            ["Cl", [ 0.48990,  0.00050,  0.05510]],
            ["F", [-0.00780,  0.94520,  1.46400]]
        ]

        with self.assertRaises(ValueError):
            Link(0, 4, factor=1., species=not_supported_chem_group)

    def test_cf3(self):
        """Test capping with CF3 (built-in)."""

        broken_link = Link(0, 4, factor=1., species="CF3")
        new_pos = broken_link.relink(xyz_ethane)

        # First CH3 + capping group.
        xyz = xyz_ethane[:4] + new_pos

        # Those position have been verified with a molecular visualization software.
        ref_capped = atom_string_to_list("""
            C     0.71660    0.89800    0.64250
            H     0.53970    1.76660   -0.00250
            H     0.48990    0.00050    0.05510
            H    -0.00780    0.94520    1.46400
            C     2.15410    0.87450    1.16590
            F     2.20191    1.41954    2.38719
            F     2.94837    1.57115    0.34441
            F     2.59308   -0.38799    1.23257
        """)

        # Every atom must be the same (same order too).
        # Position can be almost equals.
        for i, atom in enumerate(xyz):
            self.assertEqual(atom[0], ref_capped[i][0])

        # Testing only the position of the central atom in the chemical group.
        np.testing.assert_almost_equal(xyz[4][1], ref_capped[4][1], decimal=4)

    def test_custom_species(self):
        """Test capping with a custom chemical group (here -CFICl)."""

        cap = [
            ["X", [ 2.15410,  0.87450,  1.16590]],
            ["C", [ 0.71660,  0.89800,  0.64250]],
            ["I", [ 0.53970,  1.76660, -0.00250]],
            ["Cl", [ 0.48990,  0.00050,  0.05510]],
            ["F", [-0.00780,  0.94520,  1.46400]]
        ]

        broken_link = Link(0, 4, factor=1., species=cap)
        new_pos = broken_link.relink(xyz_ethane)

        # First CH3 + capping group.
        xyz = xyz_ethane[:4] + new_pos

        # Those position have been verified with a molecular visualization software.
        ref_capped = atom_string_to_list("""
            C     0.71660    0.89800    0.64250
            H     0.53970    1.76660   -0.00250
            H     0.48990    0.00050    0.05510
            H    -0.00780    0.94520    1.46400
            C     2.15410    0.87450    1.16590
            I     2.22665    0.32520    2.11184
            Cl    2.83194    0.39157    0.45228
            F     2.53182    1.88810    1.34418
        """)

        # Every atom must be the same (same order too).
        # Position can be almost equals.
        for i, atom in enumerate(xyz):
            self.assertEqual(atom[0], ref_capped[i][0])

        # Testing only the position of the central atom in the chemical group.
        np.testing.assert_almost_equal(xyz[4][1], ref_capped[4][1], decimal=4)


if __name__ == "__main__":
    unittest.main()
