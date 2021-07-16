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
        opt_dict['vqe_options'] = {'optimizer': LBFGSB_optimizer, "frozen_orbitals": 0} # on my local branch, freezing core still gives me 14 qubits, thats weird. results are not good
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        energies, operators, adapt_vqe = adapt_solver.simulate()
        print(f'ADAPT RESOURCES:\n {adapt_solver.get_resources()}\n')
        print()


if __name__ == "__main__":
    unittest.main()
