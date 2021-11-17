import unittest
import numpy as np

from agnostic_simulator import Simulator
from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g

class QMFTest(unittest.TestCase):
    def test_qmf_set_var_params(self):
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        qmf_ansatz = QMF(mol_H2_sto3g)
    
        eight_zeros = np.zeros((8,))

        qmf_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_zeros, decimal=6)

        qmf_ansatz.set_var_params([0., 0., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_zeros, decimal=6)

        eight_ones = np.ones((8,))

        qmf_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_ones, decimal=6)

        qmf_ansatz.set_var_params(np.array([1., 1., 1., 1., 1., 1., 1., 1.]))
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_ones, decimal=6)

    def test_qmf_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)

        self.assertRaises(ValueError, qmf_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_qmf_set_params_hartree_fock_state_H2(self):
        """ Verify closed-shell QMF functionalities for H2: upper case HF-State initial parameters """
        """ Only check the theta Bloch angles -- phi Bloch angles are randomly initialized """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("HF-State")

        expected = [3.141592653589793, 3.141592653589793, 0., 0.]
        self.assertAlmostEqual(np.linalg.norm(qmf_ansatz.var_params[:4]), np.linalg.norm(expected), delta=1e-10)


    def test_qmf_set_params_hartree_fock_state_H2(self):
        """ Verify closed-shell QMF functionalities for H2: lower case hf-state initial parameters """
        """ Only check the theta Bloch angles -- phi Bloch angles are randomly initialized """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("hf-state")

        expected = [3.141592653589793, 3.141592653589793, 0., 0.]
        self.assertAlmostEqual(np.linalg.norm(qmf_ansatz.var_params[:4]), np.linalg.norm(expected), delta=1e-10)

    def test_qmf_set_params_hartree_fock_state_H4(self):
        """ Verify closed-shell QMF functionalities for H4: hf-state initial parameters """
        """ Only check the theta Bloch angles -- phi Bloch angles are randomly initialized """

        qmf_ansatz = QMF(mol_H4_sto3g)
        qmf_ansatz.set_var_params("hf-state")

        expected = [3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 
                    0.,                0.,                0.,                0.]       
        self.assertAlmostEqual(np.linalg.norm(qmf_ansatz.var_params[:8]), np.linalg.norm(expected), delta=1e-10)

    def test_qmf_H2(self):
        """ Verify closed-shell QMF functionalities for H2 """

        var_params = [3.14159265, 3.14159265, 0.,         0.,
                      4.61265659, 0.73017920, 1.03851163, 2.48977533]

        # Build circuit
        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham 

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qmf_ansatz.update_var_params(var_params)
                                      
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1166843870853400, delta=1e-6)

    def test_qmf_H4_open(self):
        """ Verify open-shell QMF functionalities for H4 """

        var_params = [3.14159265, 3.14159265, 3.14159265, 0., 
                      0.,         0.,         0.,         0.,
                      2.56400050, 5.34585441, 1.46689000, 1.3119943,
                      2.95766833, 5.00079708, 3.53150391, 1.9093635]

        # Build circuit
        qmf_ansatz = QMF(mol_H4_cation_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham 

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qmf_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.5859184313544759, delta=1e-6)

    def test_qmf_H4(self):
        """ Verify closed-shell QMF functionalities for H4. """

        var_params = [3.14159265, 3.14159265, 3.14159265, 3.14159265,
                      0.,         0.,         0.,         0.,
                      0.45172480, 5.30089707, 4.52791163, 4.85272121,
                      2.28473042, 2.15616885, 0.81424786, 5.11611248]

        # Build circuit
        qmf_ansatz = QMF(mol_H4_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham 

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qmf_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.7894832518559973, delta=1e-6)

if __name__ == "__main__":
    unittest.main()
