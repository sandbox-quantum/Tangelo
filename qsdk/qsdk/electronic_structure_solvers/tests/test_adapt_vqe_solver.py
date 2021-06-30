import unittest
from pyscf import gto
import numpy as np
from agnostic_simulator import Simulator
from qsdk.electronic_structure_solvers import ADAPTSolver
from qsdk.electronic_structure_solvers.adapt_vqe_solver import LBFGSB_optimizer


H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.74137727))]
H4 = [["H", [0.7071067811865476, 0.0, 0.0]], ["H", [0.0, 0.7071067811865476, 0.0]],
      ["H", [-1.0071067811865476, 0.0, 0.0]], ["H", [0.0, -1.0071067811865476, 0.0]]]
NaH = [("Na", (0., 0., 0.)), ("H", (0., 0., 1.91439))]
H2O = """
    O    0.0000   0.0000   0.1173
    H    0.0000   0.7572  -0.4692
    H    0.0000  -0.7572  -0.4692
"""

mol_H2 = gto.Mole()
mol_H2.atom = H2
mol_H2.basis = "sto-3g"
mol_H2.charge = 0
mol_H2.spin = 0
mol_H2.build()

mol_H4 = gto.Mole()
mol_H4.atom = H4
mol_H4.basis = "sto-3g"
mol_H4.charge = 0
mol_H4.spin = 0
mol_H4.build()

mol_NaH = gto.Mole()
mol_NaH.atom = NaH
mol_NaH.basis = "sto-3g"
mol_NaH.charge = 0
mol_NaH.spin = 0
mol_NaH.build()

mol_H2O = gto.Mole()
mol_H2O.atom = H2O
mol_H2O.basis = "sto-3g"
mol_H2O.charge = 0
mol_H2O.spin = 0
mol_H2O.build()


class ADAPTSolverTest(unittest.TestCase):

    def test_instantiation_adapt(self):
        """ Try instantiating VQESolver with basic input """

        # Insert molecule here
        opt_dict = {"molecule": mol_H2O}
        opt_dict['vqe_options'] = {'optimizer': LBFGSB_optimizer, "frozen_orbitals": 0}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        energies, operators, adapt_vqe = adapt_solver.simulate()
        print(f'ADAPT RESOURCES:\n {adapt_solver.get_resources()}\n')
        print()


    # def test_simulate_h2(self):
    #     """ Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """
    #
    #     vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
    #                    "initial_var_params": [0.1, 0.1], "verbose": True}
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #
    #     energy = vqe_solver.simulate()
    #     self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)
    #
    # def test_simulate_h4(self):
    #     """ Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """
    #     vqe_options = {"molecule": mol_H4, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
    #                    "initial_var_params": "MP2", "verbose": False}
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #
    #     energy = vqe_solver.simulate()
    #     self.assertAlmostEqual(energy, -1.9778312978826869, delta=1e-4)
    #
    # def test_optimal_circuit_h4(self):
    #     """ Run VQE on H4 molecule, save optimal circuit. Verify it yields optimal energy """
    #     vqe_options = {"molecule": mol_H4, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
    #                    "initial_var_params": "MP2", "verbose": False}
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #     energy = vqe_solver.simulate()
    #
    #     sim = Simulator(target='qulacs')
    #     self.assertAlmostEqual(energy, sim.get_expectation_value(vqe_solver.qubit_hamiltonian, vqe_solver.optimal_circuit),
    #                            delta=1e-10)
    #
    # def test_custom_vqe(self):
    #     """ VQE with custom optimizer and non-optimal variational parameters """
    #
    #     # Define and assign custom optimizer: cobyla
    #     def cobyla_oneshot_optimizer(func, var_params):
    #         from scipy.optimize import minimize
    #         result = minimize(func, var_params, method="COBYLA", options={"disp": True, "maxiter": 100})
    #         return result.fun
    #
    #     vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
    #                    "initial_var_params": "ones", "verbose": False,
    #                    "optimizer": cobyla_oneshot_optimizer}
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #
    #     energy = vqe_solver.simulate()
    #     self.assertAlmostEqual(energy, -1.137270422018, places=6)
    #
    # def test_mapping_BK(self):
    #     """Test that BK mapping recovers the expected result,
    #     to within 1e-6 Ha, for the example of H2 and MP2 initial guess"""
    #     vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
    #                    "qubit_mapping": 'bk'}
    #
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #     energy = vqe_solver.simulate()
    #
    #     energy_target = -1.137270
    #     self.assertAlmostEqual(energy, energy_target, places=5)
    #
    # def test_mapping_scBK(self):
    #     """Test that scBK mapping recovers the expected result,
    #     to within 1e-6 Ha, for the example of H2 and MP2 initial guess"""
    #     vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
    #                    "qubit_mapping": 'scbk'}
    #
    #     vqe_solver = VQESolver(vqe_options)
    #     vqe_solver.build()
    #     energy = vqe_solver.simulate()
    #
    #     energy_target = -1.137270
    #     self.assertAlmostEqual(energy, energy_target, places=5)

if __name__ == "__main__":
    unittest.main()
