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

from tangelo.algorithms.variational import SA_OO_Solver, BuiltInAnsatze
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule

Li2 = [('Li', (0, 0, 0)), ('Li', (3.5, 0, 0))]
mol = SecondQuantizedMolecule(Li2, q=0, spin=0, basis="6-31g(d,p)", frozen_orbitals=[0, 1]+list(range(4, 28)))


class SA_OO_SolverTest(unittest.TestCase):

    def test_build(self):
        """Try instantiating SA_OO_Solver with basic input."""

        opt_dict = {"molecule": mol, "ref_states": [[1, 1, 0, 0], [1, 0, 1, 0]], "max_cycles": 5}
        sa_oo_solver = SA_OO_Solver(opt_dict)
        sa_oo_solver.build()

        # Test error is raised when no molecule is provided
        opt_dict = {"max_cycles": 15}
        self.assertRaises(ValueError, SA_OO_Solver, opt_dict)

    def test_li2(self):
        """Try reproducing Li2 2 electrons in 2 orbitals casscf for both states from pyscf
        """

        opt_dict = {"molecule": mol, "ref_states": [[1, 1, 0, 0], [1, 0, 1, 0]], "qubit_mapping": "bk", "up_then_down": False,
                    "tol": 1.e-5, "ansatz": BuiltInAnsatze.UCCGD, "weights": [1, 1], "n_oo_per_iter": 3}
        sa_oo_solver = SA_OO_Solver(opt_dict)
        sa_oo_solver.build()
        sa_oo_solver.iterate()

        # Code to generate exact pyscf results
        # mol = pyscf.M(atom=Li2, basis='6-31g(d,p)', spin=0)
        # myhf=mol.RHF().run()
        # weights = np.ones((2))/2
        # ncas, nelecas = (2, (1,1))
        # mycas = myhf.CASSCF(ncas, nelecas).state_average_(weights)
        # res = mycas.kernel()
        # mc = myhf.CASCI(ncas, nelecas)
        # mc.fcisolver.nroots = 2
        # exact_energies = mc.casci(mycas.mo_coeff)[0]
        exact_energies = [-14.87324, -14.85734]

        self.assertAlmostEqual(sa_oo_solver.state_energies[0], exact_energies[0], places=4)
        self.assertAlmostEqual(sa_oo_solver.state_energies[1], exact_energies[1], places=4)

        oo_resources = sa_oo_solver.get_resources()
        self.assertEqual(oo_resources["circuit_width"], 4)
        self.assertEqual(oo_resources["vqe_variational_parameters"], 3)


if __name__ == "__main__":
    unittest.main()
