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

from tangelo.molecule_library import mol_H2O_321g
from tangelo.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core, get_orbitals_excluding_homo_lumo


class FrozenOrbitalsTest(unittest.TestCase):

    def test_get_frozen_core(self):
        """Verify if the frozen orbital suggestions are consistent with chemical
        intuition.
        """

        frozen_h2o = get_frozen_core(mol_H2O_321g)
        self.assertEqual(frozen_h2o, 1)

    def test_get_homo_lumo(self):
        """Verify if the HOMO-LUMO suggestions are consistent with the provided
        parameters.
        """

        # Getting HOMO-LUMO.
        frozen_homo_lumo = get_orbitals_excluding_homo_lumo(mol_H2O_321g)
        self.assertEqual(frozen_homo_lumo, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12])

        # Active space from HOMO-2 to LUMO+4
        frozen_homo2_lumo4 = get_orbitals_excluding_homo_lumo(mol_H2O_321g, 2, 4)
        self.assertEqual(frozen_homo2_lumo4, [0, 1, 10, 11, 12])


if __name__ == "__main__":
    unittest.main()
