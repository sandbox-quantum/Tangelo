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

import numpy as np

from tangelo.algorithms import BuiltInAnsatze, SA_VQESolver
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.ansatz_generator import UCCSD
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.molecular_computation.rdms import matricize_2rdm


class SA_VQESolverTest(unittest.TestCase):

    def test_instantiation_sa_vqe(self):
        """Try instantiating SA_VQESolver with basic input."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "ref_states": [[1, 1, 0, 0]]}
        SA_VQESolver(options)

    def test_instantiation_sa_vqe_incorrect_keyword(self):
        """Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "dummy": True}
        self.assertRaises(KeyError, SA_VQESolver, options)

    def test_instantiation_sa_vqe_missing_molecule(self):
        """Instantiating with no molecule should return an error."""

        options = {"qubit_mapping": "jw"}
        self.assertRaises(ValueError, SA_VQESolver, options)

    def test_instantiation_sa_vqe_missing_ref_states(self):
        """Instantiating with no ref_states should return an error."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw"}
        self.assertRaises(ValueError, SA_VQESolver, options)

    def test_get_resources_h2_mappings(self):
        """Resource estimation, with UCCSD ansatz, given initial parameters.
        Each of JW, BK, and scBK mappings are checked.
        """
        mappings = ["jw", "bk", "scbk", "jkmn"]
        expected_values = [(15, 4), (15, 4), (5, 2), (15, 4)]

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": [0.1, 0.1], "ref_states": [[1, 1, 0, 0]]}
        for index, mi in enumerate(mappings):
            vqe_options["qubit_mapping"] = mi
            sa_vqe_solver = SA_VQESolver(vqe_options)
            sa_vqe_solver.build()
            resources = sa_vqe_solver.get_resources()

            self.assertEqual(resources["qubit_hamiltonian_terms"], expected_values[index][0])
            self.assertEqual(resources["circuit_width"], expected_values[index][1])

    def test_energy_estimation_sa_vqe(self):
        """A single SA-VQE energy evaluation for H2, using optimal parameters and
        exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "ref_states": [[1, 1, 0, 0]]}
        sa_vqe_solver = SA_VQESolver(vqe_options)
        sa_vqe_solver.build()

        energy = sa_vqe_solver.energy_estimation([5.86665842e-06, 5.65317429e-02])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_operator_expectation_sa_vqe(self):
        """ A test of the operator_expectation function, using optimal parameters and exact simulator """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": 'jw', "ref_states": [[1, 1, 0, 0]]}
        sa_vqe_solver = SA_VQESolver(vqe_options)
        sa_vqe_solver.build()

        # Test using var_params input and Qubit Hamiltonian
        energy = sa_vqe_solver.operator_expectation(sa_vqe_solver.qubit_hamiltonian, var_params=[5.86665842e-06, 5.65317429e-02],
                                                    ref_state=sa_vqe_solver.reference_circuits[0])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

        # Test using updated var_params and Fermion Hamiltonian
        sa_vqe_solver.ansatz.update_var_params([5.86665842e-06, 5.65317429e-02])
        energy = sa_vqe_solver.operator_expectation(mol_H2_sto3g.fermionic_hamiltonian, ref_state=sa_vqe_solver.reference_circuits[0])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

        # Test the three in place operators
        n_electrons = sa_vqe_solver.operator_expectation('N', ref_state=sa_vqe_solver.reference_circuits[0])
        self.assertAlmostEqual(n_electrons, 2, places=6)
        spin_z = sa_vqe_solver.operator_expectation('Sz', ref_state=sa_vqe_solver.reference_circuits[0])
        self.assertAlmostEqual(spin_z, 0, places=6)
        spin2 = sa_vqe_solver.operator_expectation('S^2', ref_state=sa_vqe_solver.reference_circuits[0])
        self.assertAlmostEqual(spin2, 0, places=6)

    def test_simulate_h2(self):
        """Run SA-VQE on H2 molecule, with UpCCGSD ansatz, JW qubit mapping, ref_states and exact simulator.
        Followed by deflation of two states calculated to determine next excited state.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UpCCGSD, "qubit_mapping": "jw",
                       "verbose": True, "ref_states": [[1, 1, 0, 0], [1, 0, 1, 0]]}
        sa_vqe_solver = SA_VQESolver(vqe_options)
        sa_vqe_solver.build()

        # code to generate exact results.
        # from pyscf import mcscf
        # mc = mcscf.CASCI(mol_H2_sto3g.mean_field, 2, 2)
        # mc.fcisolver.nroots = 3
        # exact_energies = mc.casci()[0]
        exact_energies = [-1.1372702, -0.5324790, -0.1699013]

        # Use state averaging to get ground and first excited state.
        _ = sa_vqe_solver.simulate()
        np.testing.assert_array_almost_equal(exact_energies[:2], sa_vqe_solver.state_energies, decimal=3)

        # Use deflation to get second excited state from circuits for ground state and first excited state
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UpCCGSD, "qubit_mapping": "jw",
                       "verbose": True, "deflation_circuits": [sa_vqe_solver.reference_circuits[0] + sa_vqe_solver.optimal_circuit,
                                                               sa_vqe_solver.reference_circuits[1] + sa_vqe_solver.optimal_circuit],
                       "deflation_coeff": 1.5, "ref_states": [[0, 0, 1, 1]]}
        vqe_solver_2 = SA_VQESolver(vqe_options)
        vqe_solver_2.build()

        energy = vqe_solver_2.simulate()
        self.assertAlmostEqual(exact_energies[2], energy, delta=1.e-3)

    def test_get_rdm_h2(self):
        """Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized
        parameters, exact simulator (H2).
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw", "ref_states": [[1, 1, 0, 0]]}
        sa_vqe_solver = SA_VQESolver(vqe_options)
        sa_vqe_solver.build()

        # Compute RDM matrices
        one_rdm, two_rdm = sa_vqe_solver.get_rdm([5.86665842e-06, 5.65317429e-02], ref_state=sa_vqe_solver.reference_circuits[0])

        # Test traces of matrices
        n_elec, n_orb = mol_H2_sto3g.n_active_electrons, mol_H2_sto3g.n_active_mos
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_custom_vqe(self):
        """SA-VQE with custom optimizer and non-optimal variational parameters."""

        # Define and assign custom optimizer: cobyla
        def cobyla_oneshot_optimizer(func, var_params):
            from scipy.optimize import minimize
            result = minimize(func, var_params, method="COBYLA", options={"disp": True, "maxiter": 100})
            return result.fun, result.x

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "ones", "verbose": False, "ref_states": [[1, 1, 0, 0]],
                       "optimizer": cobyla_oneshot_optimizer}
        sa_vqe_solver = SA_VQESolver(vqe_options)
        sa_vqe_solver.build()

        energy = sa_vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_qubit_qhamiltonian_input(self):
        """Test the case where a qubit Hamiltonian is used to construct SA-VQE."""

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian,
                   "ansatz": UCCSD(mol_H2_sto3g, mapping="jw"), "ref_states": [[1, 1, 0, 0]]}
        SA_VQESolver(options)

    def test_qubit_qhamiltonian_input_conflicts(self):
        """Test the case where a molecule and a qubit Hamiltonian are passed as
        inputs.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"molecule": mol_H2_sto3g,
                   "qubit_hamiltonian": qubit_hamiltonian,
                   "ansatz": UCCSD(mol_H2_sto3g, mapping="jw"), "ref_states": [[1, 1, 0, 0]]}

        with self.assertRaises(ValueError):
            SA_VQESolver(options)

    def test_qubit_qhamiltonian_input_no_custom_ansatz(self):
        """Test the case where no custom ansatz is passed when using a qubit
        Hamiltonian as input.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian, "ref_states": [[1, 1, 0, 0]]}

        with self.assertRaises(TypeError):
            SA_VQESolver(options).build()

        options["ansatz"] = BuiltInAnsatze.UCCSD

        with self.assertRaises(TypeError):
            SA_VQESolver(options).build()


if __name__ == "__main__":
    unittest.main()
