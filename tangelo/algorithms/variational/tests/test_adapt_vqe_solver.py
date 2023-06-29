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

from tangelo.molecule_library import mol_H2_sto3g, xyz_H4
from tangelo.linq import Circuit, Gate
from tangelo.algorithms.variational import ADAPTSolver
from tangelo.toolboxes.ansatz_generator.fermionic_operators import spinz_operator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize, get_qft_circuit
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator._unitary_majorana_cc import get_majorana_uccgsd_pool, get_majorana_uccsd_pool
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule


class ADAPTSolverTest(unittest.TestCase):

    def test_build_adapt(self):
        """Try instantiating ADAPTSolver with basic input."""

        opt_dict = {"molecule": mol_H2_sto3g, "max_cycles": 15, "spin": 0}
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
                     "circuit_depth": 73,
                     "circuit_2qubit_gates": 48,
                     "circuit_var_gates": 8,
                     "vqe_variational_parameters": 1}
        self.assertEqual(adapt_solver.get_resources(), resources)

    def test_multiple_cycle_adapt_majorana_pool(self):
        """Solve H4 with one frozen orbital with ADAPTSolver using 4 cycles and operators chosen
        from a Majorana UCCGSD pool and a Majorana UCCSD pool
        """

        mol = SecondQuantizedMolecule(xyz_H4, 0, 0, basis="sto-3g", frozen_orbitals=[0])
        opt_dict = {"molecule": mol, "max_cycles": 4, "verbose": False, "pool": get_majorana_uccgsd_pool,
                    "pool_args": {"n_sos": mol.n_active_sos}}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.8945, places=3)

        mol = SecondQuantizedMolecule(xyz_H4, 0, 0, basis="sto-3g", frozen_orbitals=[0])
        opt_dict = {"molecule": mol, "max_cycles": 4, "verbose": False, "pool": get_majorana_uccsd_pool,
                    "pool_args": {"n_electrons": mol.n_active_electrons, "n_sos": mol.n_active_sos}}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.8945, places=3)

        deflation_circuits = [adapt_solver.optimal_circuit]
        ref_state = [0, 1, 0, 0, 0, 1]
        opt_dict = {"molecule": mol, "max_cycles": 1, "verbose": False, "pool": get_majorana_uccgsd_pool,
                    "pool_args": {"n_sos": mol.n_active_sos},
                    "deflation_circuits": deflation_circuits, "ref_state": ref_state, "deflation_coeff": 1}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        optimal_energy = adapt_solver.simulate()

        self.assertAlmostEqual(optimal_energy, -1.91062, places=3)

    def test_multiple_cycle_adapt_majorana_pool_with_deflation(self):
        """Solve H4 with one frozen orbtial with ADAPTSolver using 4 cycles and operators chosen
        from a Majorana UCCGSD pool followed by deflation for an orthogonal state triplet state.
        """

        mol = SecondQuantizedMolecule(xyz_H4, 0, 0, basis="sto-3g", frozen_orbitals=[0])
        opt_dict = {"molecule": mol, "max_cycles": 4, "verbose": False, "pool": get_majorana_uccgsd_pool,
                    "pool_args": {"n_sos": mol.n_active_sos}}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.8945, places=3)

        deflation_circuits = [adapt_solver.optimal_circuit]
        ref_state = [0, 1, 0, 0, 0, 1]
        opt_dict = {"molecule": mol, "max_cycles": 1, "verbose": False, "pool": get_majorana_uccgsd_pool,
                    "pool_args": {"n_sos": mol.n_active_sos},
                    "deflation_circuits": deflation_circuits, "ref_state": ref_state, "deflation_coeff": 1}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        optimal_energy = adapt_solver.simulate()

        self.assertAlmostEqual(optimal_energy, -1.91062, places=3)

    def test_sym_projected(self):
        """Solve Linear H3 with one frozen orbtial with ADAPTSolver using 4 cycles and operators chosen
        from a Majorana UCCGSD pool, with a sz symmetry projection.
        """

        xyz_H3 = [("H", [0., 0., 0.]), ("H", [1., 0., 0.]), ("H", [2., 0., 0.])]
        mol = SecondQuantizedMolecule(xyz_H3, 0, 1, basis="sto-3g")

        # QPE based Sz projection.
        def sz_check(n_state: int, molecule: SecondQuantizedMolecule, mapping: str, up_then_down):
            n_qft = 3
            spin_fe_op = spinz_operator(molecule.n_active_mos)
            q_spin = fermion_to_qubit_mapping(spin_fe_op, mapping, molecule.n_active_sos, molecule.n_active_electrons, up_then_down, molecule.spin)

            sym_var_circuit = Circuit([Gate("H", n_state + q) for q in range(n_qft)])
            for j, i in enumerate(range(n_state, n_state+n_qft)):
                sym_var_circuit += trotterize(2*q_spin+3, -2*np.pi/2**(j+1), control=i)
            sym_var_circuit += get_qft_circuit(list(range(n_state+n_qft-1, n_state-1, -1)), inverse=True)
            sym_var_circuit += Circuit([Gate("MEASURE", i) for i in range(n_state, n_state+n_qft)])
            return sym_var_circuit

        proj_circuit = sz_check(6, mol, "JW", False)

        opt_dict = {"molecule": mol, "max_cycles": 4, "verbose": False, "pool": get_majorana_uccgsd_pool,
                    "pool_args": {"n_sos": mol.n_active_sos}, "simulate_options": {"desired_meas_result": "100"},
                    "projective_circuit": proj_circuit}
        adapt_solver = ADAPTSolver(opt_dict)
        adapt_solver.build()
        adapt_solver.simulate()

        self.assertAlmostEqual(adapt_solver.optimal_energy, -1.56835, places=5)


if __name__ == "__main__":
    unittest.main()
