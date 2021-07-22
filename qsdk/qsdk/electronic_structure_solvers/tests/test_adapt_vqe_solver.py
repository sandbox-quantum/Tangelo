import unittest
from pyscf import gto
from qsdk.electronic_structure_solvers import ADAPTSolver

LiH = [("Li", (0., 0., 0.)), ("H", (0., 0., 1.91439))]
mol_LiH = gto.Mole()
mol_LiH.atom = LiH
mol_LiH.basis = "sto-3g"
mol_LiH.charge = 0
mol_LiH.spin = 0
mol_LiH.build()


class ADAPTSolverTest(unittest.TestCase):

    def test_instantiation_adapt(self):
        """ Try instantiating ADAPTSolver with basic input """

        # Insert molecule here
        opt_dict = {"molecule": mol_LiH, "frozen_orbitals": 0, "max_cycles": 1}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        #energies = adapt_solver.simulate()
        #print(f'ADAPT RESOURCES:\n {adapt_solver.get_resources()}\n')

    def test_instantiation_adapt(self):
        """ Try instantiating ADAPTSolver with basic input """

        # Insert molecule here
        opt_dict = {"molecule": mol_LiH, "frozen_orbitals": 0, "max_cycles": 1}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        #energies = adapt_solver.simulate()
        #print(f'ADAPT RESOURCES:\n {adapt_solver.get_resources()}\n')


if __name__ == "__main__":
    unittest.main()
