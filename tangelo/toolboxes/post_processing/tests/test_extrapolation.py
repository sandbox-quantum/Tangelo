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

from tangelo.toolboxes.post_processing import diis, richardson

energies = [-1.04775574, -1.04302289, -1.03364568, -1.03005245]
coeffs = [1., 1.1, 1.2, 1.3]


class ExtrapolationTest(unittest.TestCase):

    def test_diis(self):
        """Test DIIS extrapolation on small sample data
        """
        calculated = diis(energies, coeffs)
        self.assertAlmostEqual(-1.11047933, calculated, delta=1e-6)

    def test_richardson(self):
        """Test Richardson extrapolation on small sample data
        """
        calculated = richardson(energies, coeffs)
        self.assertAlmostEqual(-1.45459036, calculated, delta=1e-6)
        calculated = richardson(energies, coeffs, estimate_exp=True)
        self.assertAlmostEqual(-1.05601603, calculated, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
