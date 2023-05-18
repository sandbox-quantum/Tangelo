import unittest

import numpy as np

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.toolboxes.molecular_computation.integral_solver_psi4 import IntegralSolver_psi4
from tangelo.algorithms.variational import SA_OO_Solver, BuiltInAnsatze, ADAPTSolver
from tangelo.molecule_library import xyz_H4, mol_H4_minao

h2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]


class Testpsi4(unittest.TestCase):

    def test_sa_oo_vqe(self):
        "Test that sa_oo_vqe works properly when using a user defined IntegralSolver that only reads in integrals"
        molecule_dummy = SecondQuantizedMolecule(h2, 0, 0, IntegralSolver_psi4(), basis="6-31g", frozen_orbitals=[3])
        sa_oo_vqe = SA_OO_Solver({"molecule": molecule_dummy, "ref_states": [[1, 1, 0, 0, 0, 0]],
                                  "tol": 1.e-5, "ansatz": BuiltInAnsatze.UCCSD, "n_oo_per_iter": 25,
                                  "initial_var_params": [1.e-5]*5})
        sa_oo_vqe.build()
        sa_oo_vqe.iterate()

        self.assertAlmostEqual(sa_oo_vqe.state_energies[0], -1.15137, places=4)

    def test_adapt_vqe_solver(self):
        "Test that ADAPTVQE works with a user defined IntegralSolver."
        molecule_dummy = SecondQuantizedMolecule(h2, 0, 0, IntegralSolver_psi4(), basis="6-31g", frozen_orbitals=[])

        adapt_vqe = ADAPTSolver({"molecule": molecule_dummy})
        adapt_vqe.build()
        energy = adapt_vqe.simulate()
        self.assertAlmostEqual(energy, -1.15168, places=4)

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
        self.assertAlmostEqual(-1.901623, e_oniom_vqe, places=5)
        self.assertEqual(mol_H4_minao, None)


if __name__ == "__main__":
    unittest.main()
