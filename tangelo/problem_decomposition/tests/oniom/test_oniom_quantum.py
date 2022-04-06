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

        options_both = {"basis": "sto-3g"}

        # With this line, the interaction between H2-H2 is computed with a low
        # accuracy method.
        system = Fragment(solver_low="HF", options_low=options_both)
        # VQE-UCCSD fragments.
        model_vqe_1 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
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
        model_vqe_1 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="QITE",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="QITE",
                               options_high=options_both,
                               selected_atoms=[2, 3])
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        e_oniom_vqe = oniom_model_vqe.simulate()

        # ONIOM + CCSD is tested in test_oniom.ONIOMTest.test_energy_hf_ccsd_h4.
        self.assertAlmostEqual(-1.901616, e_oniom_vqe, places=5)

    def test_get_resources(self):
        """Test to verifiy the implementation of resources estimation in ONIOM."""

        options_both = {"basis": "sto-3g"}

        system = Fragment(solver_low="HF", options_low=options_both)

        # VQE-UCCSD fragments.
        model_vqe_1 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="HF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
                               selected_atoms=[2, 3])
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        vqe_resources = {"qubit_hamiltonian_terms": 15,
                         "circuit_width": 4,
                         "circuit_gates": 158,
                         "circuit_2qubit_gates": 64,
                         "circuit_var_gates": 12,
                         "vqe_variational_parameters": 2}

        res = oniom_model_vqe.get_resources()

        self.assertEqual(res[1], vqe_resources)
        self.assertEqual(res[2], vqe_resources)


if __name__ == "__main__":
    unittest.main()
