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

from tangelo import SecondQuantizedMolecule
from tangelo.helpers.utils import installed_chem_backends
from tangelo.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.toolboxes.molecular_computation.integral_solver_psi4 import IntegralSolverPsi4
from tangelo.algorithms.variational import SA_OO_Solver, BuiltInAnsatze, ADAPTSolver, iQCC_solver
from tangelo.molecule_library import xyz_H4, mol_H4_minao, xyz_H2, mol_H4_sto3g_uhf_a1_frozen


@unittest.skipIf("psi4" not in installed_chem_backends, "Test Skipped: Backend not available \n")
class Testpsi4(unittest.TestCase):

    def test_sa_oo_vqe(self):
        "Test that sa_oo_vqe works properly when using a IntegralSolverPsi4"
        molecule_dummy = SecondQuantizedMolecule(xyz_H2, 0, 0, IntegralSolverPsi4(), basis="6-31g", frozen_orbitals=[3])
        sa_oo_vqe = SA_OO_Solver({"molecule": molecule_dummy, "ref_states": [[1, 1, 0, 0, 0, 0]],
                                  "tol": 1.e-5, "ansatz": BuiltInAnsatze.UCCSD, "n_oo_per_iter": 25,
                                  "initial_var_params": [1.e-5]*5})
        sa_oo_vqe.build()
        sa_oo_vqe.iterate()

        self.assertAlmostEqual(sa_oo_vqe.state_energies[0], -1.15137, places=4)

    def test_adapt_vqe_solver(self):
        "Test that ADAPT-VQE works with IntegralSolverPsi4."
        molecule_dummy = SecondQuantizedMolecule(xyz_H2, 0, 0, IntegralSolverPsi4(), basis="6-31g", frozen_orbitals=[])

        adapt_vqe = ADAPTSolver({"molecule": molecule_dummy})
        adapt_vqe.build()
        energy = adapt_vqe.simulate()
        self.assertAlmostEqual(energy, -1.15168, places=4)

    def test_energy_hf_vqe_uccsd_h4(self):
        """Test psi4 with HF and VQE (with UCCSD) in ONIOM."""

        options_hf = {"basis": "sto-3g"}
        options_vqe = {"basis": "sto-3g", "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "ones"}

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
        self.assertAlmostEqual(-1.901623, e_oniom_vqe, places=5)
        self.assertEqual(mol_H4_minao, None)

    def test_iqcc_h4_uhf(self):
        """Test the energy after 3 iterations for H4 uhf with 1 alpha orbital frozen and generators limited to 8"""

        ansatz_options = {"max_qcc_gens": 8}

        iqcc_options = {"molecule": mol_H4_sto3g_uhf_a1_frozen,
                        "qubit_mapping": "scbk",
                        "up_then_down": True,
                        "ansatz_options": ansatz_options,
                        "deqcc_thresh": 1e-5,
                        "max_iqcc_iter": 3}

        iqcc = iQCC_solver(iqcc_options)
        iqcc.build()
        iqcc_energy = iqcc.simulate()

        self.assertAlmostEqual(iqcc_energy, -1.95831, places=3)


if __name__ == "__main__":
    unittest.main()
