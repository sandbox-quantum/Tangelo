import unittest
import numpy as np
from pyscf import gto

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.qubit_mappings import jordan_wigner
from qsdk.toolboxes.ansatz_generator.hea import HEA

from agnostic_simulator import Simulator

# Initiate simulator
sim = Simulator(target="qulacs")

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


class HEATest(unittest.TestCase):

    def test_hea_set_var_params(self):
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array)."""

        molecule = MolecularData(mol_h2)
        hea_ansatz = HEA({'molecule': molecule})

        hea_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.ones(4 * 3 * 3), decimal=6)

        hea_ansatz.set_var_params(np.ones(4 * 3 * 3))
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.ones(4 * 3 * 3), decimal=6)

        hea_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.zeros(4 * 3 * 3), decimal=6)

    def test_hea_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """
        molecule = MolecularData(mol_h2)
        hea_ansatz = HEA({'molecule': molecule})

        self.assertRaises(ValueError, hea_ansatz.set_var_params, np.ones(4 * 3 * 3 + 1))

    def test_hea_H2(self):
        """ Verify closed-shell HEA functionalities for H2 """

        molecule = MolecularData(mol_h2)

        # Build circuit
        hea_ansatz = HEA({'molecule': molecule})
        hea_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        params = [ 1.96262489e+00, -9.83505909e-01, -1.92659543e+00,  3.68638855e+00,
                   4.72852133e+00,  4.50012102e+00,  4.71213972e+00, -4.72825044e+00,
                   4.46356300e+00, -3.14524440e+00,  4.71239665e+00,  1.79676072e+00,
                   4.16741524e+00, -3.06056499e+00, -2.70616213e+00, -2.05962953e+00,
                   6.39944906e+00, -1.44020337e+00, -6.33177816e+00,  3.14216799e+00,
                  -2.94623607e+00,  3.15592219e+00, -5.14396016e+00,  1.08194211e+00,
                   6.25913105e-01, -7.62290954e-02,  3.52590185e-03, -1.57161049e+00,
                   1.55418991e+00, -3.14115924e+00,  4.69079147e+00,  1.57141235e+00,
                  -2.32267456e+00,  3.26312961e+00, -2.72130709e+00, -1.55068880e+00]

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1371208680826537, delta=1e-6)

    def test_hea_H4(self):
        """ Verify closed-shell HEA functionalities for H4 """

        molecule = MolecularData(mol_h4)
        var_params = [-3.48779066e-01,  9.13702020e-01,  2.74069489e-02,  6.28325477e+00,
                       1.46828618e+00, -3.75087033e-03, -1.50679387e+00,  4.48185703e+00,
                       8.01901597e-03, -4.80582521e+00,  1.49877501e+00,  1.57103183e+00,
                      -1.56739221e+00, -4.70926775e+00,  3.72882771e+00, -3.12566508e+00,
                       6.27777542e+00,  4.76764774e+00, -3.79502797e+00, -4.70972512e+00,
                       1.54411852e+00, -1.25781487e-01, -4.12791444e+00, -1.60220361e+00]

        # Build circuit with Ry rotationas instead of RZ*RX*RZ rotations
        hea_ansatz = HEA({'molecule': molecule, 'rot_type': 'real'})
        hea_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9217382203638278, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
