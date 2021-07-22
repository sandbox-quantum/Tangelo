import unittest
from pyscf import gto
from qsdk.electronic_structure_solvers import ADAPTSolver

H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.75))]
mol_H2 = gto.Mole()
mol_H2.atom = H2
mol_H2.basis = "sto-3g"
mol_H2.charge = 0
mol_H2.spin = 0
mol_H2.build()


class ADAPTSolverTest(unittest.TestCase):

    def test_build_adapt(self):
        """Try instantiating ADAPTSolver with basic input. """

        opt_dict = {"molecule": mol_H2, "frozen_orbitals": 0, "max_cycles": 15}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()

    def test_single_cycle_adapt(self):
        """Try instantiating ADAPTSolver with basic input. The fermionic term
        ordering has been taken from the reference below (original paper for
        ADAPT-VQE).

        Ref
            Grimsley, H.R., Economou, S.E., Barnes, E. et al.
            An adaptive variational algorithm for exact molecular simulations on a quantum computer.
            Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2
        """

        opt_dict = {"molecule": mol_H2, "frozen_orbitals": 0, "max_cycles": 1, "verbose": False}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.137117, places=3)

        resources = {"qubit_hamiltonian_terms": 15,
                     "circuit_width": 4,
                     "circuit_gates": 122,
                     "circuit_2qubit_gates": 48,
                     "circuit_var_gates": 8,
                     "vqe_variational_parameters": 1}
        self.assertEqual(adapt_solver.get_resources(), resources)


if __name__ == "__main__":
    unittest.main()
