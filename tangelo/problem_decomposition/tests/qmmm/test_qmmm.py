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

import os
import unittest

from tangelo.algorithms.variational import BuiltInAnsatze
from tangelo.problem_decomposition.qmmm.qmmm_problem_decomposition import QMMMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment, Link

pwd_this_test = os.path.dirname(os.path.abspath(__file__))+"/"


class ONIOMTest(unittest.TestCase):

    def test_energy_fci_hf(self):
        """Testing the QM/MM energy with a HF molecule and an partial charge of -0.3 at (0.5, 0.6, 0.8) at FCI level
        """

        options_both = {"basis": "sto-3g"}
        geometry = [("H", (0, 0, 0)), ("F", (0, 0, 1))]
        charges = [-0.3]
        coords = [(0.5, 0.6, 0.8)]

        system = Fragment(solver_high="fci", options_low=options_both)
        qmmm_model_cc = QMMMProblemDecomposition({"geometry": geometry, "qmfragment": system, "charges": charges, "coords": coords})

        e_qmmm_cc = qmmm_model_cc.simulate()
        self.assertAlmostEqual(-98.62087, e_qmmm_cc, places=4)

    def test_energy_ccsd_ala_ala_ala(self):
        """Test that the QM/MM energy is correct when a pdb file is provided and indices are selected as qm region using Fragment"""

        frag = Fragment(solver_high="ccsd", selected_atoms=[6, 7, 8, 9], broken_links=[Link(6, 4, factor=0.71)],
                        options_high={"basis": "sto-3g"})
        qmmm = QMMMProblemDecomposition({"geometry": pwd_this_test+"ala_ala_ala.pdb", "qmfragment": frag, "mmpackage": "rdkit"})

        energy = qmmm.simulate()

        self.assertAlmostEqual(-39.67720, energy, delta=1.e-4)

    def test_energy_fci_h2_ala_ala_ala(self):
        """Test that the reference energy is returned when an H2 QM geometry is placed next to a pdb charges with VQE as the solver"""

        qmmm_h2 = QMMMProblemDecomposition({"geometry": [("H", (-2, 0, 0)), ("H", (-2, 0, 1))], "charges": [pwd_this_test+"ala_ala_ala.pdb"],
                                            "mmpackage": "rdkit",
                                            "qmfragment": Fragment(solver_high="vqe", options_high={"basis": "sto-3g", "ansatz": BuiltInAnsatze.QCC,
                                                                                                    "up_then_down": True})})
        energy = qmmm_h2.simulate()
        self.assertAlmostEqual(-1.10258, energy, delta=1.e-4)
        self.assertEqual(qmmm_h2.get_resources()["qubit_hamiltonian_terms"], 27)


if __name__ == "__main__":
    unittest.main()
