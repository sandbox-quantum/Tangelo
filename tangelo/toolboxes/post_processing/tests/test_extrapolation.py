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
import os
import numpy as np

from tangelo.toolboxes.post_processing import diis, richardson

path_data = os.path.dirname(os.path.abspath(__file__)) + "/data"


class ExtrapolationTest(unittest.TestCase):

    def test_diis(self):
        """Test DIIS extrapolation on small sample data from Alejandro
        """
        with open(f"{path_data}/diis_test.dat") as f:
            data = np.loadtxt(f)
            coeffs = data[:-1, 0]
            energies = data[:-1, 1]
            reference = data[-1, 1]

        calculated = diis(energies, coeffs)
        self.assertAlmostEqual(reference, calculated, delta=1e-10)

    def test_richardson(self):
        """Test Richardson extrapolation on small sample data from Alejandro
        """
        with open(f"{path_data}/richardson_test.dat") as f:
            data = np.loadtxt(f)
            coeffs = data[:-1, 0]
            energies = data[:-1, 1]
            reference = data[-1, 1]

        calculated = richardson(energies, coeffs)
        self.assertAlmostEqual(reference, calculated, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
