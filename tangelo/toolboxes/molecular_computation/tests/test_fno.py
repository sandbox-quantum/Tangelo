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

import numpy as np

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2_321g, mol_H4_cation_sto3g
from tangelo.toolboxes.molecular_computation.fno import FNO


class FNOTest(unittest.TestCase):

    def test_h2_321g_fno_restricted(self):
        """Test the FNO active space selection and the molecular coefficients
        for the H2/3-21G molecular system, with a restricted mean-field.
        """

        fno = FNO(mol_H2_321g, 0.50)
        frozen_orbitals = fno.get_frozen_indices()

        self.assertEqual([2, 3], frozen_orbitals)

        fno_mo_coeff_ref = np.array([
            [0.2896, -0.6979, -0.7861, -1.0728],
            [0.3121, -0.6255, 0.6627, 1.2638],
            [0.2896, 0.6979, -0.7861, 1.0728],
            [0.3121, 0.6255, 0.6627, -1.2638]
        ])

        # Testing the absolute values, because phase changes can occur randomly.
        np.testing.assert_array_almost_equal(np.abs(fno_mo_coeff_ref), np.abs(fno.mo_coeff), decimal=4)

    def test_hf_ccpvdz_fno_restricted(self):
        """Test the FNO active space selection for the CF/cc-pvDZ molecular
        system, with a restricted mean-field.
        """

        xyz_HF = """
            H 0.0000 0.0000 0.0000
            F 0.0000 0.0000 0.9168
        """
        mol = SecondQuantizedMolecule(xyz_HF, 0, 0, basis="cc-pvdz", uhf=False, frozen_orbitals=[1, 2])
        fno = FNO(mol, 0.98)
        frozen_orbitals = fno.get_frozen_indices()

        ref_frozen_orbitals = [1, 2, 13, 14, 15, 16, 17, 18]

        self.assertEqual(ref_frozen_orbitals, frozen_orbitals)

    def test_cf2_ccpvdz_fno_unrestricted(self):
        """Test the FNO active space selection for the CF2/cc-pvDZ molecular
        system, with an unrestricted mean-field.
        """

        xyz_CF2 = """
            C 0.0000 0.0000 0.5932
            F 0.0000 1.0282 -0.1977
            F 0.0000 -1.0282 -0.1977
        """
        mol = SecondQuantizedMolecule(xyz_CF2, 0, 2, basis="cc-pvdz", uhf=True, frozen_orbitals=10)
        fno = FNO(mol, 0.4)
        frozen_orbitals = fno.get_frozen_indices()

        ref_frozen_orbitals = [
            [i for i in range(42) if i not in {10, 11, 12, 13, 14}],
            [i for i in range(42) if i not in {10, 11}]
        ]

        self.assertEqual(ref_frozen_orbitals, frozen_orbitals)

    def test_h2_invalid_threshold(self):
        """Test the threshold rejection mechanism. """

        with self.assertRaises(ValueError):
            FNO(mol_H2_321g, 1.1)

        with self.assertRaises(ValueError):
            FNO(mol_H2_321g, [1., 1.])

    def test_h4_cation_sto3g_rohf_notimplemented(self):
        """Test if the NotImplementedError is raises with a ROHF molecule. """

        with self.assertRaises(NotImplementedError):
            FNO(mol_H4_cation_sto3g, 0.5)


if __name__ == "__main__":
    unittest.main()
