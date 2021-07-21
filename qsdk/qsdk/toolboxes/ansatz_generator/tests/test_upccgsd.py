import unittest
import numpy as np
from pyscf import gto

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.qubit_mappings import jordan_wigner
from qsdk.toolboxes.ansatz_generator.upccgsd import UpCCGSD

from agnostic_simulator import Simulator

# Build molecule objects used by the tests
H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
H4 = [['H', [0.7071067811865476, 0.0, 0.0]], ['H', [0.0, 0.7071067811865476, 0.0]],
      ['H', [-1.0071067811865476, 0.0, 0.0]], ['H', [0.0, -1.0071067811865476, 0.0]]]

mol_h2 = gto.Mole()
mol_h2.atom = H2
mol_h2.basis = "sto-3g"
mol_h2.spin = 0
mol_h2.build()

mol_h4 = gto.Mole()
mol_h4.atom = H4
mol_h4.basis = "sto-3g"
mol_h4.spin = 0
mol_h4.build()

mol_h4_open = gto.Mole()
mol_h4_open.atom = H4
mol_h4_open.basis = "sto-3g"
mol_h4_open.charge = 1
mol_h4_open.spin = 1
mol_h4_open.build()


class UpCCGSDTest(unittest.TestCase):

    def test_upccgsd_set_var_params(self):
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array).
        """

        molecule = MolecularData(mol_h2)
        upccgsd_ansatz = UpCCGSD(molecule)

        upccgsd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, np.array([1., 1., 1., 1., 1., 1.]), decimal=6)

        upccgsd_ansatz.set_var_params([1., 1., 1., 1., 1., 1.])
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, np.array([1., 1., 1., 1., 1., 1.]), decimal=6)

        upccgsd_ansatz.set_var_params(np.array([1., 1., 1., 1., 1., 1.]))
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, np.array([1., 1., 1., 1., 1., 1.]), decimal=6)

    def test_upccgsd_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """
        molecule = MolecularData(mol_h2)
        upccgsd_ansatz = UpCCGSD(molecule)

        self.assertRaises(ValueError, upccgsd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_upccgsd_H2(self):
        """ Verify closed-shell UpCCGSD functionalities for H2 """

        molecule = MolecularData(mol_h2)

        # Build circuit
        upccgsd_ansatz = UpCCGSD(molecule)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        upccgsd_ansatz.update_var_params([0.03518165, -0.02986551,  0.02897598, -0.03632711,
                                          0.03044071,  0.08252277])
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372658, delta=1e-6)

    def test_upccgsd_H4_open(self):
        """ Verify open-shell UpCCGSD functionalities for H4 """

        molecule = MolecularData(mol_h4_open)
        var_params = [-0.0291763,   0.36927821,  0.14654907, -0.13845063,  0.14387348, -0.00903457,
                      -0.56843484,  0.01223853,  0.13649942,  0.83225887,  0.20236275,  0.02682977,
                      -0.17198068,  0.10161518,  0.01523924,  0.30848876,  0.22430705, -0.07290468,
                       0.16253591,  0.02268874,  0.2382988,   0.33716289, -0.20094664,  0.3057071,
                      -0.58426117,  0.22433297,  0.29668267,  0.64761217, -0.2705204,   0.07540534,
                       0.20131878,  0.09890588,  0.10563459, -0.22983007,  0.13578206, -0.02017009]

        # Build circuit
        upccgsd_ansatz = UpCCGSD(molecule)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        upccgsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.6412047312, delta=1e-6)

    def test_upccgsd_H4(self):
        """ Verify closed-shell UpCCGSD functionalities for H4 """

        molecule = MolecularData(mol_h4)
        var_params = [ 3.79061344e-02, -4.30067212e-02,  3.02230152e-02,  3.42936301e-03,
                      -6.09234584e-04, -3.14370905e-02, -4.86666676e-02,  5.16834522e-02,
                       2.58779710e-03,  2.58848760e-03,  2.50477121e-02,  3.13929977e-02,
                       6.60326773e-03, -9.12896032e-02,  1.53572944e-01,  1.87098400e-01,
                      -4.15148608e-02,  4.92466084e-02,  7.04965743e-01,  7.18739139e-01,
                       1.54329908e-02, -3.33233433e-02, -9.08825509e-03,  3.93555394e-02,
                       5.47661674e-02, -4.91387503e-02, -1.11946468e-02, -2.61401420e-02,
                       5.38915256e-01,  5.37080003e-01, -1.77856374e-02,  1.17461439e-02,
                       2.73552144e-02,  6.02186436e-01, -8.58400153e-02, -1.17667425e-01]

        # Build circuit
        upccgsd_ansatz = UpCCGSD(molecule)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        upccgsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.968345618, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
