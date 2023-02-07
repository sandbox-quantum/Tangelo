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
from openfermion.chem.molecular_data import spinorb_from_spatial

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.molecular_computation.coefficients import spatial_from_spinorb


class CoefficientsTest(unittest.TestCase):

    def test_spatial_from_spinorb(self):
        """Test the conversion from spinorbitals to MO coefficients."""
        _, one_body_mos, two_body_mos = mol_H2_sto3g.get_integrals()

        one_body_sos, two_body_sos = spinorb_from_spatial(one_body_mos, two_body_mos)
        one_body_mos_recomputed, two_body_mos_recomputed = spatial_from_spinorb(one_body_sos, two_body_sos)

        np.testing.assert_array_almost_equal(one_body_mos, one_body_mos_recomputed)
        np.testing.assert_array_almost_equal(two_body_mos, two_body_mos_recomputed)


if __name__ == "__main__":
    unittest.main()
