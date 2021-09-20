import unittest
import numpy as np

from agnostic_simulator import Simulator
from qsdk.electronic_structure_solvers import BuiltInAnsatze, VQESolver
from qsdk.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g, mol_NaH_sto3g, mol_NaH_sto3g
from qsdk.toolboxes.ansatz_generator.uccsd import UCCSD
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


def matricize_2rdm(two_rdm, n_orbitals):
    """Turns the two_rdm tensor into a matrix for test purposes."""

    l = 0
    sq = n_orbitals * n_orbitals
    jpqrs = np.zeros((n_orbitals, n_orbitals), dtype=int)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            jpqrs[i, j] = l
            l += 1

    rho = np.zeros((sq, sq), dtype=complex)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            ij = jpqrs[i, j]
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    kl = jpqrs[k, l]
                    rho[ij, kl] += two_rdm[i, k, j, l]
    return rho


class VQESolverTest(unittest.TestCase):

    def test_instantiation_vqe(self):
        """Try instantiating VQESolver with basic input."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw"}
        VQESolver(options)

    def test_instantiation_vqe_incorrect_keyword(self):
        """Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "dummy": True}
        self.assertRaises(KeyError, VQESolver, options)

    def test_instantiation_vqe_missing_molecule(self):
        """Instantiating with no molecule should return an error."""

        options = {"qubit_mapping": "jw"}
        self.assertRaises(ValueError, VQESolver, options)

    def test_get_resources_h2_mappings(self):
        """Resource estimation, with UCCSD ansatz, given initial parameters.
        Each of JW, BK, and scBK mappings are checked.
        """
        mappings = ["jw", "bk", "scbk"]
        expected_values = [(15, 4), (15, 4), (5, 2)]

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": [0.1, 0.1]}
        for index, mi in enumerate(mappings):
            vqe_options["qubit_mapping"] = mi
            vqe_solver = VQESolver(vqe_options)
            vqe_solver.build()
            resources = vqe_solver.get_resources()

            self.assertEqual(resources["qubit_hamiltonian_terms"], expected_values[index][0])
            self.assertEqual(resources["circuit_width"], expected_values[index][1])

    def test_energy_estimation_vqe(self):
        """A single VQE energy evaluation for H2, using optimal parameters and
        exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.energy_estimation([5.86665842e-06, 5.65317429e-02])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_simulate_h2(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": [0.1, 0.1], "verbose": True}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_simulate_h2_qiskit(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact qiskit simulator.
        """

        backend_options = {"target": "qiskit", "n_shots": None, "noise_model": None}
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": [6.28531447e-06, 5.65431626e-02], "verbose": True,
                       "backend_options": backend_options}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        self.assertAlmostEqual(energy, -1.13727042117, delta=1e-6)

    def test_simulate_h4(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """
        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.9778312978826869, delta=1e-4)

    def test_simulate_h4_open(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """
        vqe_options = {"molecule": mol_H4_cation_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "random", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.6394, delta=1e-3)

    def test_optimal_circuit_h4(self):
        """Run VQE on H4 molecule, save optimal circuit. Verify it yields
        optimal energy.
        """
        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        sim = Simulator(target="qulacs")
        self.assertAlmostEqual(energy, sim.get_expectation_value(vqe_solver.qubit_hamiltonian, vqe_solver.optimal_circuit),
                               delta=1e-10)

    def test_get_rdm_h2(self):
        """Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized
        parameters, exact simulator (H2).
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        one_rdm, two_rdm = vqe_solver.get_rdm([5.86665842e-06, 5.65317429e-02])

        # Test traces of matrices
        n_elec, n_orb = mol_H2_sto3g.n_active_electrons, mol_H2_sto3g.n_active_mos
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_get_rdm_h4(self):
        """Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized
        parameters, exact simulator (H4).
        """

        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        var_params = [-6.47627367e-06, -5.24257363e-06, -5.99540594e-06, -7.70205325e-06, 1.15628926e-02,
                      3.42313563e-01,  3.48211343e-02,  1.49150233e-02, 7.53406401e-02,  8.44095525e-03,
                      -1.79981377e-01, -1.00585201e-01, 1.02162534e-02, -3.65870070e-02]
        one_rdm, two_rdm = vqe_solver.get_rdm(var_params)
        # Test traces of matrices
        n_elec, n_orb = mol_H4_sto3g.n_active_electrons, mol_H4_sto3g.n_active_mos
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_custom_vqe(self):
        """VQE with custom optimizer and non-optimal variational parameters."""

        # Define and assign custom optimizer: cobyla
        def cobyla_oneshot_optimizer(func, var_params):
            from scipy.optimize import minimize
            result = minimize(func, var_params, method="COBYLA", options={"disp": True, "maxiter": 100})
            return result.fun, result.x

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "ones", "verbose": False,
                       "optimizer": cobyla_oneshot_optimizer}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_mapping_BK(self):
        """Test that BK mapping recovers the expected result, to within 1e-6 Ha,
        for the example of H2 and MP2 initial guess.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
                       "qubit_mapping": "bk"}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_scBK(self):
        """Test that scBK mapping recovers the expected result, to within
        1e-6 Ha, for the example of H2 and MP2 initial guess.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
                       "qubit_mapping": "scbk"}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_spin_reorder_equivalence(self):
        """Test that re-ordered spin input (all up followed by all down) returns
        the same optimized energy result for both JW and BK mappings.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "up_then_down": True,
                       "verbose": False, "qubit_mapping": "jw"}

        vqe_solver_jw = VQESolver(vqe_options)
        vqe_solver_jw.build()
        energy_jw = vqe_solver_jw.simulate()

        vqe_options["qubit_mapping"] = "bk"
        vqe_solver_bk = VQESolver(vqe_options)
        vqe_solver_bk.build()
        energy_bk = vqe_solver_bk.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy_jw, energy_target, places=5)
        self.assertAlmostEqual(energy_bk, energy_target, places=5)

    def test_simulate_h4_frozen_orbitals(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator. First (occupied) and last (virtual)
        orbitals are frozen.
        """
        mol_H4_sto3g_frozen = mol_H4_sto3g.freeze_mos([0, 3], inplace=False)

        vqe_options = {"molecule": mol_H4_sto3g_frozen, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.8943598012229799, delta=1e-5)

    def test_simulate_nah_rucc(self):
        """Run VQE on NaH molecule, with UCC1 and UCC3 ansatze, JW qubit
        mapping. The computation is mapped to a HOMO-LUMO problem.
        """

        mol_NaH_sto3g_2mos = mol_NaH_sto3g.freeze_mos([i for i in range(9) if i not in [5, 9]], inplace=False)

        vqe_options = {"molecule": mol_NaH_sto3g_2mos, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "jw",
                       "initial_var_params": "zeros", "up_then_down": True, "verbose": False}

        vqe_solver_ucc1 = VQESolver(vqe_options)
        vqe_solver_ucc1.build()
        energy_ucc1 = vqe_solver_ucc1.simulate()

        vqe_options["ansatz"] = BuiltInAnsatze.UCC3
        vqe_solver_ucc3 = VQESolver(vqe_options)
        vqe_solver_ucc3.build()
        energy_ucc3 = vqe_solver_ucc3.simulate()

        self.assertAlmostEqual(energy_ucc1, -160.30334365109297, delta=1e-6)
        self.assertAlmostEqual(energy_ucc3, -160.30345935884606, delta=1e-6)

    def test_toomany_orbitals_rucc(self):
        """Test the case where there is too many orbitals in the system to be
        mapped into a HOMO-LUMO problem.
        """

        vqe_options = {"molecule": mol_NaH_sto3g, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "jw",
                       "initial_var_params": "zeros", "up_then_down": True, "verbose": False}

        with self.assertRaises(ValueError):
            vqe_solver_ucc1 = VQESolver(vqe_options)
            vqe_solver_ucc1.build()

        with self.assertRaises(ValueError):
            vqe_options["ansatz"] = BuiltInAnsatze.UCC3
            vqe_solver_ucc3 = VQESolver(vqe_options)
            vqe_solver_ucc3.build()

    def test_wrong_mapping_rucc(self):
        """Test the case where another mapping process (not JW) is selected."""

        mol_NaH_sto3g_2mos = mol_NaH_sto3g.freeze_mos([i for i in range(9) if i not in [5, 9]], inplace=False)

        vqe_options = {"molecule": mol_NaH_sto3g_2mos, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "bk",
                       "initial_var_params": "zeros", "up_then_down": True}

        with self.assertRaises(ValueError):
            vqe_solver_ucc1 = VQESolver(vqe_options)
            vqe_solver_ucc1.build()

        with self.assertRaises(ValueError):
            vqe_options["ansatz"] = BuiltInAnsatze.UCC3
            vqe_solver_ucc3 = VQESolver(vqe_options)
            vqe_solver_ucc3.build()

    def test_qubit_qhamiltonian_input(self):
        """Test the case where a qubit Hamiltonian is used to construct VQE."""

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian,
                   "ansatz": UCCSD(mol_H2_sto3g, mapping="jw")}
        VQESolver(options)

    def test_qubit_qhamiltonian_input_conflicts(self):
        """Test the case where a molecule and a qubit Hamiltonian are passed as
        inputs.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"molecule": mol_H2_sto3g,
                   "qubit_hamiltonian": qubit_hamiltonian,
                   "ansatz": UCCSD(mol_H2_sto3g, mapping="jw")}

        with self.assertRaises(ValueError):
            VQESolver(options)

    def test_qubit_qhamiltonian_input_no_custom_ansatz(self):
        """Test the case where no custom ansatz is passed when using a qubit
        Hamiltonian as input.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian}

        with self.assertRaises(TypeError):
            VQESolver(options).build()

        options["ansatz"] = BuiltInAnsatze.UCCSD

        with self.assertRaises(TypeError):
            VQESolver(options).build()


if __name__ == "__main__":
    unittest.main()
