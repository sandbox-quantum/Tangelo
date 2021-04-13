import unittest
from pyscf import gto
import numpy as np

from qsdk.electronic_structure_solvers.vqe_solver import Ansatze, VQESolver


H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.74137727))]
H4 = [["H", [0.7071067811865476, 0.0, 0.0]], ["H", [0.0, 0.7071067811865476, 0.0]],
      ["H", [-1.0071067811865476, 0.0, 0.0]], ["H", [0.0, -1.0071067811865476, 0.0]]]

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


def matricize_2rdm(two_rdm, n_orbitals):
    """ Turns the two_rdm tensor into a matrix for test purposes """

    l = 0
    sq = n_orbitals * n_orbitals
    jpqrs = np.zeros((n_orbitals, n_orbitals), dtype=np.int)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            jpqrs[i, j] = l
            l += 1

    rho = np.zeros((sq, sq))
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
        """ Try instantiating VQESolver with basic input """

        options = {"molecule": mol_H2, "qubit_mapping": 'jw'}
        VQESolver(options)

    def test_instantiation_vqe_incorrect_keyword(self):
        """ Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2, "qubit_mapping": 'jw', 'dummy': True}
        self.assertRaises(KeyError, VQESolver, options)

    def test_instantiation_vqe_missing_molecule(self):
        """ Instantiating with no molecule should return an error """

        options = {"qubit_mapping": 'jw'}
        self.assertRaises(ValueError, VQESolver, options)

    def test_get_resources_h2_mappings(self):
        """ Resource estimation, with UCCSD ansatz, given initial parameters.
        Each of JW, BK, and scBK mappings are checked."""
        mappings = ['jw', 'bk', 'scbk']
        expected_values = [(15, 4, 158, 64, 12, 2),(15, 4, 107, 46, 12, 2), (5, 2, 20, 4, 4, 2)]
        expected_keys = ['qubit_hamiltonian_terms', 'circuit_width', 'circuit_gates',
                    'circuit_2qubit_gates', 'circuit_var_gates', 'vqe_variational_parameters']
        
        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
                       "initial_var_params": [0.1, 0.1]}
        for mi in range(3):
            vqe_options['qubit_mapping'] = mappings[mi]      
            vqe_solver = VQESolver(vqe_options)
            vqe_solver.build()
            resources = vqe_solver.get_resources()
            print(resources)

            expected_resources = {key:expected_values[mi][vi] for vi,key in enumerate(expected_keys)}
            self.assertDictEqual(resources, expected_resources)

    def test_energy_estimation_vqe(self):
        """ A single VQE energy evaluation for H2, using optimal parameters and exact simulator """

        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw'}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.energy_estimation([5.86665842e-06, 5.65317429e-02])
        print(energy)
        self.assertAlmostEqual(energy, -1.137270422018, places=7)

    def test_simulate_h2(self):
        """ Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """

        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
                       "initial_var_params": [0.1, 0.1], "verbose": True}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_simulate_h4(self):
        """ Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """
        vqe_options = {"molecule": mol_H4, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
                       "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.9778312978826869, delta=1e-4)

    def test_get_rdm_h2(self):
        """ Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized parameters, exact simulator (H2) """

        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw'}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        one_rdm, two_rdm = vqe_solver.get_rdm([5.86665842e-06, 5.65317429e-02])

        # Test traces of matrices
        n_elec, n_orb = mol_H2.nelectron, mol_H2.nao_nr()
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_get_rdm_h4(self):
        """ Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized parameters, exact simulator (H4) """

        vqe_options = {"molecule": mol_H4, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw'}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        var_params = [-6.47627367e-06, -5.24257363e-06, -5.99540594e-06, -7.70205325e-06, 1.15628926e-02,
                      3.42313563e-01,  3.48211343e-02,  1.49150233e-02, 7.53406401e-02,  8.44095525e-03,
                      -1.79981377e-01, -1.00585201e-01, 1.02162534e-02, -3.65870070e-02]
        one_rdm, two_rdm = vqe_solver.get_rdm(var_params)
        # Test traces of matrices
        n_elec, n_orb = mol_H4.nelectron, mol_H4.nao_nr()
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_custom_vqe(self):
        """ VQE with custom optimizer and non-optimal variational parameters """

        # Define and assign custom optimizer: cobyla
        def cobyla_oneshot_optimizer(func, var_params):
            from scipy.optimize import minimize
            result = minimize(func, var_params, method="COBYLA", options={"disp": True, "maxiter": 100})
            return result.fun

        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD, "qubit_mapping": 'jw',
                       "initial_var_params": "ones", "verbose": False,
                       "optimizer": cobyla_oneshot_optimizer}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, places=7)

    def test_mapping_equivalence(self):
        """Test that JW, BK and scBK mappings all recover same result,
        to within 1e-6 Ha, for the example of H2 and MP2 initial guess"""
        vqe_options = {"molecule": mol_H2, "ansatz": Ansatze.UCCSD,
                       "initial_var_params": "MP2", "verbose": False}

        vqe_options[ "qubit_mapping"] = 'jw'
        vqe_solver_jw = VQESolver(vqe_options)
        vqe_solver_jw.build()
        energy_jw = vqe_solver_jw.simulate()

        vqe_options[ "qubit_mapping"] = 'bk'
        vqe_solver_bk = VQESolver(vqe_options)
        vqe_solver_bk.build()
        energy_bk = vqe_solver_bk.simulate()

        vqe_options[ "qubit_mapping"] = 'scbk'
        vqe_solver_scbk = VQESolver(vqe_options)
        vqe_solver_scbk.build()
        energy_scbk = vqe_solver_scbk.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy_jw, energy_target, places=5)
        self.assertAlmostEqual(energy_bk, energy_target, places=5)
        self.assertAlmostEqual(energy_scbk, energy_target, places=5)



if __name__ == "__main__":
    unittest.main()
