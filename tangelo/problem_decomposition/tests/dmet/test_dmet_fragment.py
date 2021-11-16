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

"""Test the construction of orbital list (from atom list) for DMET calculation.
"""

import unittest
from pyscf import gto

from tangelo.problem_decomposition.dmet._helpers.dmet_fragment import dmet_fragment_constructor


class TestFragments(unittest.TestCase):
    """Generate the orbital list."""

    def test_orbital_list_construction(self):

        # Initialize Molecule object with PySCF and input
        mol = gto.Mole()
        mol.atom = """
            C 0.94764 -0.02227  0.05901
            H 0.58322  0.35937 -0.89984
            H 0.54862  0.61702  0.85300
            H 0.54780 -1.03196  0.19694
            C 2.46782 -0.03097  0.07887
            H 2.83564  0.98716 -0.09384
            H 2.83464 -0.65291 -0.74596
            C 3.00694 -0.55965  1.40773
            H 2.63295 -1.57673  1.57731
            H 2.63329  0.06314  2.22967
            C 4.53625 -0.56666  1.42449
            H 4.91031  0.45032  1.25453
            H 4.90978 -1.19011  0.60302
            C 5.07544 -1.09527  2.75473
            H 4.70164 -2.11240  2.92450
            H 4.70170 -0.47206  3.57629
            C 6.60476 -1.10212  2.77147
            H 6.97868 -0.08532  2.60009
            H 6.97839 -1.72629  1.95057
            C 7.14410 -1.62861  4.10112
            H 6.77776 -2.64712  4.27473
            H 6.77598 -1.00636  4.92513
            C 8.66428 -1.63508  4.12154
            H 9.06449 -2.27473  3.32841
            H 9.02896 -2.01514  5.08095
            H 9.06273 -0.62500  3.98256
        """

        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        # Determine the number of atoms for each fragment
        fragment_atom = [4, 3, 3, 3, 3, 3, 3, 4]

        # Test the construction of orbitals lists
        orb_list, orb_list2, atom_list2 = dmet_fragment_constructor(mol, fragment_atom, 1)
        self.assertEqual(atom_list2, [7, 6, 6, 7], "The orbital list (number per fragment) does not agree")
        self.assertEqual(orb_list, [28, 26, 26, 28], "The orbital list (number per fragment) does not agree")
        self.assertEqual(orb_list2, [[0, 28], [28, 54], [54, 80], [80, 108]], "The min max list does not agree")


if __name__ == "__main__":
    unittest.main()
