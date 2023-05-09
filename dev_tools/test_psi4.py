import unittest

import numpy as np

from tangelo import SecondQuantizedMolecule
from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver
from tangelo.toolboxes.molecular_computation.integral_solver_psi4 import IntegralSolver_psi4
from tangelo.algorithms.variational import SA_OO_Solver, BuiltInAnsatze, ADAPTSolver

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


if __name__ == "__main__":
    unittest.main()
