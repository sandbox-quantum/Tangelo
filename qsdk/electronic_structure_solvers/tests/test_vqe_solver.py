import unittest
from pyscf import gto
import numpy as np
from qsdk.electronic_structure_solvers.vqe_solver import Ansatze, VQESolver
#from qemist.util import deserialize

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

    def test_energy_estimation_vqe(self):
        """ A single VQE energy evaluation for H2, using given variational parameters
         and hardware backend """

        solver = VQESolver()
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.build(mol_H2)

        energy = solver.energy_estimation([5.86665842e-06, 5.65317429e-02])
        self.assertAlmostEqual(energy, -1.137270422018, places=7)

    def test_custom_vqe(self):
        """ VQE H2 minimal basis with custom optimizer and initial variational parameters
         Cobyla with maxiter=0 returns after evaluating energy once (for the initial var params) """

        solver = VQESolver()
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.initial_var_params = [5.86665842e-06, 5.65317429e-02]

        # Define and assign custom optimizer: cobyla
        def cobyla_oneshot_optimizer(func, var_params):
            from scipy.optimize import minimize
            result = minimize(func, var_params, method="COBYLA",
                              options={"disp": True, "maxiter": 0})
            return result.fun

        solver.optimizer = cobyla_oneshot_optimizer
        solver.build(mol_H2)

        energy = solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, places=7)

    def test_get_rdm_h2(self):
        """ VQE H2 minimal basis with custom optimizer and initial amplitudes
         Cobyla with maxiter=0 returns after evaluating energy once (for the initial var params) """

        solver = VQESolver()
        solver.verbose = False
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.build(mol_H2)

        # Compute RDM matrices
        one_rdm, two_rdm = solver.get_rdm([5.86665842e-06, 5.65317429e-02])
        # Test traces of matrices
        n_elec, n_orb = mol_H2.nelectron, mol_H2.nao_nr()
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_get_rdm_h4(self):
        """ VQE H4 minimal basis with custom optimizer and initial amplitudes
         Cobyla with maxiter=0 returns after evaluating energy once (for the initial var params) """

        solver = VQESolver()
        solver.verbose = False
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.build(mol_H4)

        # Compute RDM matrices
        var_params = [-6.47627367e-06, -5.24257363e-06, -5.99540594e-06, -7.70205325e-06, 1.15628926e-02,
                      3.42313563e-01,  3.48211343e-02,  1.49150233e-02, 7.53406401e-02,  8.44095525e-03,
                      -1.79981377e-01, -1.00585201e-01, 1.02162534e-02, -3.65870070e-02]
        one_rdm, two_rdm = solver.get_rdm(var_params)
        # Test traces of matrices
        n_elec, n_orb = mol_H4.nelectron, mol_H4.nao_nr()
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                               delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_h2_no_mf_slsqp(self):
        """ VQE-UCCSD closed-shell convergence for H2 minimal basis """
        solver = VQESolver()
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.build(mol_H2)

        energy = solver.simulate()
        self.assertAlmostEqual(energy, -1.1372704178510415, places=7)

    def test_h4_no_mf_slsqp(self):
        """ VQE-UCCSD closed-shell convergence for H4 minimal basis """
        solver = VQESolver()
        solver.backend_parameters = {"target": "qulacs"}
        solver.ansatz_type = Ansatze.UCCSD
        solver.build(mol_H4)

        energy = solver.simulate()
        self.assertAlmostEqual(energy, -1.9778312978826869, delta=1e-4)

    # @unittest.skipUnless(import_successed, "agnostic_simulator not installed")
    # def test_serialize_load_serial(self):
    #     """ Test that serializiation then loading returns equivalent values.
    #     Special focus on loading complex functional arguments set as the
    #     optimizer, backend types, and anzatz types."""
    #
    #     from functools import partial
    #     from scipy.optimize import minimize
    #
    #     optimizer = partial(minimize, method="SLSQP",
    #                         options={"disp": True, "maxiter": 2000, "eps": 1e-5, "ftol": 1e-5})
    #
    #     solver = VQESolver()
    #     solver.optimizer = optimizer
    #     solver.backend_parameters = {"target": "qulacs"}
    #     solver.ansatz_type = Ansatze.UCCSD
    #     solver.verbose = False
    #
    #     serialized = solver.serialize()
    #     solver2 = deserialize(serialized["next_solver"], serialized["solver_params"])
    #
    #     self.assertEqual(solver2.verbose, False)
    #     self.assertEqual(solver2.backend_parameters, {"target": "qulacs"})
    #     self.assertEqual(solver2.ansatz_type, Ansatze.UCCSD)
    #     # It is hard to check equality of this function, but the default init is None
    #     self.assertTrue(solver2.optimizer)


if __name__ == "__main__":
    unittest.main()