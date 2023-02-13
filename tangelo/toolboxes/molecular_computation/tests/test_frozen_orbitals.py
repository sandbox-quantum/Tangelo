# Copyright 2023 Good Chemistry Company.
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

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2O_321g, mol_H2O_sto3g
import tangelo.toolboxes.molecular_computation.frozen_orbitals as fo

mol_H2O_sto3g_uhf = SecondQuantizedMolecule(mol_H2O_sto3g.xyz, uhf=True)


class FrozenOrbitalsTest(unittest.TestCase):

    def test_get_frozen_core(self):
        """Verify if the frozen orbital suggestions are consistent with chemical
        intuition.
        """

        frozen_h2o = fo.get_frozen_core(mol_H2O_321g)
        self.assertEqual(frozen_h2o, 1)

    def test_get_homo_lumo(self):
        """Verify if the HOMO-LUMO suggestions are consistent with the provided
        parameters.
        """

        # Getting HOMO-LUMO.
        frozen_homo_lumo = fo.get_orbitals_excluding_homo_lumo(mol_H2O_321g)
        self.assertEqual(frozen_homo_lumo, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12])

        # Active space from HOMO-2 to LUMO+4
        frozen_homo2_lumo4 = fo.get_orbitals_excluding_homo_lumo(mol_H2O_321g, 2, 4)
        self.assertEqual(frozen_homo2_lumo4, [0, 1, 10, 11, 12])

    def test_freezing_orbitals_none(self):
        """Verify freezing orbitals functionalities (with None)."""
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g, None)
        self.assertEqual([0, 1, 2, 3, 4], active_occ)
        self.assertEqual([5, 6], active_virt)
        self.assertEqual([], frozen_occ)
        self.assertEqual([], frozen_virt)

    def test_freezing_orbitals_empty(self):
        """Verify freezing orbitals functionalities (empty list)."""
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g, [])
        self.assertEqual([0, 1, 2, 3, 4], active_occ)
        self.assertEqual([5, 6], active_virt)
        self.assertEqual([], frozen_occ)
        self.assertEqual([], frozen_virt)

    def test_freezing_orbitals_uhf_empty(self):
        """Verify freezing orbitals empty input."""
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g_uhf, None)
        self.assertEqual([[0, 1, 2, 3, 4]]*2, active_occ)
        self.assertEqual([[5, 6]]*2, active_virt)
        self.assertEqual([[]]*2, frozen_occ)
        self.assertEqual([[]]*2, frozen_virt)

    def test_freezing_orbitals_int(self):
        """Verify freezing orbitals functionalities (single integer)."""
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g, 3)
        self.assertEqual([3, 4], active_occ)
        self.assertEqual([5, 6], active_virt)
        self.assertEqual([0, 1, 2], frozen_occ)
        self.assertEqual([], frozen_virt)

    def test_freezing_orbitals_list_of_int(self):
        """Verify freezing orbitals functionalities (list of integers)."""
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g, [0, 1, 2, 6])
        self.assertEqual([3, 4], active_occ)
        self.assertEqual([5], active_virt)
        self.assertEqual([0, 1, 2], frozen_occ)
        self.assertEqual([6], frozen_virt)

    def test_freezing_orbitals_uhf_list_of_int(self):
        """Verify freezing orbitals functionalities for an UHF molecule (list of
        integers).
        """
        active_occ, frozen_occ, active_virt, frozen_virt = fo.convert_frozen_orbitals(mol_H2O_sto3g_uhf, [[0, 1, 2, 6], [0, 1, 2]])
        self.assertEqual([[3, 4], [3, 4]], active_occ)
        self.assertEqual([[5], [5, 6]], active_virt)
        self.assertEqual([[0, 1, 2], [0, 1, 2]], frozen_occ)
        self.assertEqual([[6], []], frozen_virt)

    def test_freezing_type_exception(self):
        """Verify freezing orbitals exceptions."""

        # Cases where the input is wrong type.
        with self.assertRaises(TypeError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, "3")
        with self.assertRaises(TypeError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, 3.141592)
        with self.assertRaises(TypeError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, [0, 1, 2.2222, 3, 4, 5])
        with self.assertRaises(TypeError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g_uhf, [[0, 1, 2.2222, 3, 4, 5]]*2)

    def test_no_active_electron(self):
        """Verify if freezing all active orbitals fails."""

        # Cases where no active electron are remaining.
        with self.assertRaises(ValueError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, 5)
        with self.assertRaises(ValueError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, [0, 1, 2, 3, 4, 5])

    def test_no_active_virtual(self):
        """Verify if freezing all virtual orbitals fails."""

        # Cases where no active virtual orbitals are remaining.
        with self.assertRaises(ValueError):
            fo.convert_frozen_orbitals(mol_H2O_sto3g, [5, 6])


if __name__ == "__main__":
    unittest.main()
