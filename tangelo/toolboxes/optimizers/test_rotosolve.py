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

from tangelo.toolboxes.optimizers.roto import rotosolve
from tangelo.algorithms import BuiltInAnsatze, VQESolver
from tangelo.molecule_library import mol_H2_sto3g


class OptimizerTest(unittest.TestCase):

    def test_VQE_rotosolve(self):
        """Run VQE on H2 molecule, using Rotosolve, with UCC3 ansatz,
        JW qubit mapping, and exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCC3,
                       "qubit_mapping": "jw", "verbose": False,
                       "optimizer": rotosolve, "up_then_down": True}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        vqe_solver.simulate()
        energy = vqe_solver.optimal_energy

        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
