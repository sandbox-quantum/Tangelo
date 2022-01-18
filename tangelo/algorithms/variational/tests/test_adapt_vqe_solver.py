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

from tangelo.algorithms.variational import ADAPTSolver
from tangelo.molecule_library import mol_H2_sto3g, xyz_H4
from tangelo.toolboxes.ansatz_generator._unitary_majorana_cc import majorana_uccgsd_list
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule


class ADAPTSolverTest(unittest.TestCase):

    def test_build_adapt(self):
        """Try instantiating ADAPTSolver with basic input."""

        opt_dict = {"molecule": mol_H2_sto3g, "max_cycles": 15}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()

    def test_single_cycle_adapt(self):
        """Try instantiating ADAPTSolver with basic input. The fermionic term
        ordering has been taken from the reference below (original paper for
        ADAPT-VQE).

        Reference:
            - Grimsley, H.R., Economou, S.E., Barnes, E. et al.
            An adaptive variational algorithm for exact molecular simulations on
            a quantum computer. Nat Commun 10, 3007 (2019).
            https://doi.org/10.1038/s41467-019-10988-2
        """

        opt_dict = {"molecule": mol_H2_sto3g, "max_cycles": 1, "verbose": False}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.13727, places=3)

        resources = {"qubit_hamiltonian_terms": 15,
                     "circuit_width": 4,
                     "circuit_gates": 122,
                     "circuit_2qubit_gates": 48,
                     "circuit_var_gates": 8,
                     "vqe_variational_parameters": 1}
        self.assertEqual(adapt_solver.get_resources(), resources)

    def test_multiple_cycle_adapt_majorana_pool(self):
        """Solve H4 with one frozen orbtial with ADAPTSolver using 4 cycles and a operators chosen
        from a Majorana UCCGSD pool
        """

        mol = SecondQuantizedMolecule(xyz_H4, 0, 0, "sto-3g", frozen_orbitals=[0])
        opt_dict = {"molecule": mol, "max_cycles": 4, "verbose": False, "pool": majorana_uccgsd_list,
                    "pool_args": {"n_sos": mol.n_active_sos}}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.8945, places=3)


if __name__ == "__main__":
    unittest.main()
