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

from tangelo.toolboxes.post_processing import diis, richardson, extrapolation

energies = [-1.1070830819357105, -1.0778342538877541, -1.0494855002828576, -1.0220085207923948,
            -0.995365932747342, -0.9695424717692709, -0.9445011607426314]
errors = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
coeffs = [1, 2, 3, 4, 5, 6, 7]


class ExtrapolationTest(unittest.TestCase):

    def test_diis(self):
        """Test DIIS extrapolation on small sample data"""
        diis_ref = [-1.1357318603549604, -1.13499594848339, -1.1341334673708405, -1.1331451262769456, -1.1320385453585242]
        err_ref = [0.024944382578492966, 0.02449489742783177, 0.02481934729198171, 0.025438378704451894, 0.026186146828319073]
        for n, ref, eref in zip(range(3, 8), diis_ref, err_ref):
            calc, err = diis(coeffs[:n], energies[:n], errors[:n])
            self.assertAlmostEqual(ref, calc, delta=1e-6)
            self.assertAlmostEqual(eref, err, delta=1e-4)

    def test_richardson(self):
        """Test Richardson extrapolation on small sample data"""
        rich_ref = [-1.1372319844267267, -1.1372602847553528, -1.137251202414955, -1.137220001783962, -1.1371449701252567]
        err_ref = [0.07348469228349534, 0.17888543819998318, 0.4183300132670378, 0.9524704719832527, 2.127815781499893]
        for n, ref, eref in zip(range(3, 8), rich_ref, err_ref):
            calc, err = richardson(coeffs[:n], energies[:n], errors[:n])
            self.assertAlmostEqual(ref, calc, delta=1e-6)
            self.assertAlmostEqual(eref, err, delta=1e-4)
            extr, erre = extrapolation(coeffs[:n], energies[:n], errors[:n])
            self.assertAlmostEqual(ref, extr, delta=1e-6)
            self.assertAlmostEqual(eref, erre, delta=1e-4)

    def test_richardson_exp(self):
        """Test Richardson extrapolation with exponent estimation on small sample data"""
        rich_ref = [-1.1168326912850293, -1.1216325249654286, -1.1222201155004157, -1.1297496614161582, -1.1689623909539615]
        err_ref = [0.014907119849998597, 0.023110652702992267, 0.04009450743442095, 0.1353768232960827, 0.35052067817606436]
        for n, ref, erref in zip(range(3, 8), rich_ref, err_ref):
            calc, err = richardson(coeffs[:n], energies[:n], errors, estimate_exp=True)
            self.assertAlmostEqual(ref, calc, delta=1e-6)
            self.assertAlmostEqual(erref, err, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
