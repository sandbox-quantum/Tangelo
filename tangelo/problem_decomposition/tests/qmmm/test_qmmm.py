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

from numpy import linspace

from tangelo.problem_decomposition.qmmm.qmmm_problem_decomposition import QMMMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment, Link


class ONIOMTest(unittest.TestCase):

    def test_energy_ccsd_hf(self):
        """Testing the oniom energy with a HF molecule and an partial charge of -0.3 at (0.5, 0.6, 0.8)
        """

        options_both = {"basis": "sto-3g"}
        geometry = [("H", (0, 0, 0)), ("F", (0, 0, 1))]
        charges = [-0.3]
        coords = [(0.5, 0.6, 0.8)]

        system = Fragment(solver_high="ccsd", options_low=options_both)
        qmmm_model_cc = QMMMProblemDecomposition({"geometry": geometry, "qmfragment": system, "charges": charges, "coords": coords})

        e_qmmm_cc = qmmm_model_cc.simulate()
        self.assertAlmostEqual(-98.62087, e_qmmm_cc, places=4)


if __name__ == "__main__":
    unittest.main()
