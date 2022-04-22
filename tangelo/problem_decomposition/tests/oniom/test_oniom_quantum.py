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

from tangelo.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.molecule_library import xyz_H4
from tangelo.algorithms.variational import BuiltInAnsatze


class ONIOMQuantumTest(unittest.TestCase):

    def test_energy_hf_vqe_uccsd_h4(self):
        """Test to verifiy the implementation of VQE (with UCCSD) in ONIOM."""

        options_hf = {"basis": "sto-3g"}
        options_vqe = {"basis": "sto-3g", "ansatz": BuiltInAnsatze.UCCSD}

        # With this line, the interaction between H2-H2 is computed with a low
        # accuracy method.
        system = Fragment(solver_low="HF", options_low=options_hf)
        # VQE-UCCSD fragments.
        model_vqe_1 = Fragment(solver_low="HF",
                               options_low=options_hf,
                               solver_high="VQE",
                               options_high=options_vqe,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="HF",
                               options_low=options_hf,
                               solver_high="VQE",
                               options_high=options_vqe,
                               selected_atoms=[2, 3])
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        e_oniom_vqe = oniom_model_vqe.simulate()

        # ONIOM + CCSD is tested in test_oniom.ONIOMTest.test_energy_hf_ccsd_h4.
        self.assertAlmostEqual(-1.901616, e_oniom_vqe, places=5)

    def test_energy_hf_qite_h4(self):
        """Test to verifiy the implementation of QITE in ONIOM."""

        options_both = {"basis": "sto-3g", "verbose": False}

        # With this line, the interaction between H2-H2 is computed with a low
        # accuracy method.
        system = Fragment(solver_low="HF", options_low=options_both)
        # QITE fragments.
        model_qite_1 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="QITE",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_qite_2 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="QITE",
                               options_high=options_both,
                               selected_atoms=[2, 3])
        oniom_model_qite = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_qite_1, model_qite_2]})

        e_oniom_qite = oniom_model_qite.simulate()

        # ONIOM + CCSD is tested in test_oniom.ONIOMTest.test_energy_hf_ccsd_h4.
        self.assertAlmostEqual(-1.901616, e_oniom_qite, places=5)

    def test_energy_hf_adapt_h4(self):
        """Test to verifiy the implementation of ADAPT-VQE in ONIOM."""

        options_both = {"basis": "sto-3g", "verbose": False}

        # With this line, the interaction between H2-H2 is computed with a low
        # accuracy method.
        system = Fragment(solver_low="HF", options_low=options_both)
        # ADAPT fragments.
        model_adapt_1 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="ADAPT",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_adapt_2 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="ADAPT",
                               options_high=options_both,
                               selected_atoms=[2, 3])
        oniom_model_adapt = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_adapt_1, model_adapt_2]})

        e_oniom_adapt = oniom_model_adapt.simulate()

        # ONIOM + CCSD is tested in test_oniom.ONIOMTest.test_energy_hf_ccsd_h4.
        self.assertAlmostEqual(-1.901616, e_oniom_adapt, places=5)

    def test_get_resources_vqe(self):
        """Test to verifiy the implementation of resources estimation (VQE) in
        ONIOM. Other quantum solvers should also work if VQE works.
        """

        options_hf = {"basis": "sto-3g"}
        options_vqe = {"basis": "sto-3g", "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "JW"}

        system = Fragment(solver_low="HF", options_low=options_hf)
        # VQE-UCCSD fragments.
        model_vqe_1 = Fragment(solver_low="HF",
                               options_low=options_hf,
                               solver_high="VQE",
                               options_high=options_vqe,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="HF",
                               options_low=options_hf,
                               solver_high="VQE",
                               options_high=options_vqe,
                               selected_atoms=[2, 3])
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        res = oniom_model_vqe.get_resources()

        self.assertEqual(res[1]["circuit_width"], 4)
        self.assertEqual(res[1]["qubit_hamiltonian_terms"], 15)

        self.assertEqual(res[2]["circuit_width"], 4)
        self.assertEqual(res[2]["qubit_hamiltonian_terms"], 15)


if __name__ == "__main__":
    unittest.main()
