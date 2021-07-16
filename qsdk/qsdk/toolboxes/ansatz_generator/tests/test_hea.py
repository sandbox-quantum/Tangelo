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

        params = [ 1.96262489e+00, -9.83505909e-01, -1.92659544e+00,  3.68638855e+00,
                   4.71410736e+00,  4.78247991e+00,  4.71258582e+00, -4.79077006e+00,
                   4.60613188e+00, -3.14130503e+00,  4.71232383e+00,  1.35715841e+00,
                   4.30998410e+00, -3.00415626e+00, -1.06784872e+00, -2.05119893e+00,
                   6.44114344e+00, -1.56358255e+00, -6.28254779e+00,  3.14118427e+00,
                  -3.10505551e+00,  3.15123780e+00, -3.64794717e+00,  1.09127829e+00,
                   4.67093656e-01, -1.19912860e-01,  3.99351728e-03, -1.57104046e+00,
                   1.56811666e+00, -3.14050540e+00,  4.71181097e+00,  1.57036595e+00,
                  -2.16414405e+00,  3.40295404e+00, -2.87986715e+00, -1.69054279e+00]

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372661564779496, delta=1e-6)

    def test_hea_H4(self):
        """ Verify closed-shell HEA functionalities for H4 """

        molecule = MolecularData(mol_h4)
        var_params = [-2.34142720e-04,  6.28472420e+00,  4.67668267e+00, -3.14063369e+00,
                      -1.53697174e+00, -6.22546556e+00, -3.11351342e+00,  3.14158366e+00,
                      -1.57812617e+00,  3.14254101e+00, -6.29656957e+00, -8.93646210e+00,
                       7.84465163e-04, -1.32569792e-05,  1.90710480e+00,  6.44924149e+00,
                      -4.69797158e+00,  3.47688734e+00,  8.24677008e-04,  1.89006312e-04,
                      -1.57734737e+00, -3.63191375e+00, -3.15706332e+00, -1.73625091e+00]

        # Build circuit with Ry rotationas instead of RZ*RX*RZ rotations
        hea_ansatz = HEA({'molecule': molecule, 'rot_type': 'real'})
        hea_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.6534450968231997, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
