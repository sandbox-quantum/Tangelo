# Copyright SandboxAQ 2021-2024.
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
from tangelo.problem_decomposition import DMETProblemDecomposition
from tangelo.problem_decomposition.dmet import Localization

LiO2 = """
    Li              0.000000    0.000000    1.380605
    O               0.000000    0.676045   -0.258863
    O               0.000000   -0.676045   -0.258863
"""


class OSDMETProblemDecompositionTest(unittest.TestCase):

    def test_lio2_sto6g_restricted_ccsd(self):
        """Tests the result from OS-DMET (ROHF) against a value from a reference
        implementation with nao localization and CCSD solution to fragments.
        """

        mol_lio2 = SecondQuantizedMolecule(LiO2, q=0, spin=1, basis="STO-6G", frozen_orbitals=None, uhf=False)

        opt_dmet = {"molecule": mol_lio2,
                    "fragment_atoms": [1, 1, 1],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.nao,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -156.6317605935, places=4)

    def test_lio2_sto6g_unrestricted_ccsd(self):
        """Tests the result from OS-DMET (UHF) against a value from a reference
        implementation with nao localization and CCSD solution to fragments.
        """

        mol_lio2 = SecondQuantizedMolecule(LiO2, q=0, spin=1, basis="STO-6G", frozen_orbitals=None, uhf=True)

        opt_dmet = {"molecule": mol_lio2,
                    "fragment_atoms": [1, 1, 1],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.nao,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -156.6243118102, places=4)

    def test_lio2_sto6g_restricted_hf(self):
        """Tests the result from a single loop of OS-DMET (ROHF) against a value
        with nao localization and HF solution to fragments."""

        mol_lio2 = SecondQuantizedMolecule(LiO2, q=0, spin=1, basis="STO-6G", frozen_orbitals=None, uhf=False)

        opt_dmet = {"molecule": mol_lio2,
                    "fragment_atoms": [1, 1, 1],
                    "fragment_solvers": "hf",
                    "electron_localization": Localization.nao,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        dmet_solver._oneshot_loop(0.)
        energy = dmet_solver.dmet_energy

        # Not sure on how to validate this case.
        self.assertAlmostEqual(energy, -156.34040, places=4)

    def test_lio2_sto6g_unrestricted_hf(self):
        """Tests the result from a single loop of OS-DMET (UHF) against a value
        with nao localization and HF solution to fragments."""

        mol_lio2 = SecondQuantizedMolecule(LiO2, q=0, spin=1, basis="STO-6G", frozen_orbitals=None, uhf=True)

        opt_dmet = {"molecule": mol_lio2,
                    "fragment_atoms": [1, 1, 1],
                    "fragment_solvers": "hf",
                    "electron_localization": Localization.nao,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        dmet_solver._oneshot_loop(0.)
        energy = dmet_solver.dmet_energy

        self.assertAlmostEqual(energy, mol_lio2.mf_energy, places=2)


if __name__ == "__main__":
    unittest.main()
