import unittest
import numpy as np

from tangelo.backendbuddy import Simulator
from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g

class QCCTest(unittest.TestCase):

    def test_qcc_set_var_params(self):
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        qcc_ansatz = QCC(mol_H2_sto3g)
    
        one_zero = np.zeros((1,))

        qcc_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, one_zero, decimal=6)

        qcc_ansatz.set_var_params([0.])
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, one_zero, decimal=6)

        one_one = np.ones((1,))

        qcc_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, one_one, decimal=6)

        qcc_ansatz.set_var_params(np.array([1.]))
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, one_one, decimal=6)

    def test_qcc_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        qcc_ansatz = QCC(mol_H2_sto3g)

        self.assertRaises(ValueError, qcc_ansatz.set_var_params, np.array([1., 1.]))

    def test_qmf_qcc_H2(self):
        """ Verify closed-shell QMF + QCC functionalities for H2 """

        # Determine the QMF parameter set and cicuit for H2
        qmf_ansatz = QMF(mol_H2_sto3g, up_then_down=True)
        qmf_ansatz.build_circuit()

        # Build the QCC class, passing the QMF parameters and circuit
        qcc_op_list = [QubitOperator("X0 Y1 Y2 Y3")]
        qcc_var_params = [0.2261362655507856]
        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True, qubit_op_list=qcc_op_list, qmf_var_params=qmf_ansatz.var_params, qmf_circuit=qmf_ansatz.circuit)

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham 

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)
                                      
        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701746609022, delta=1e-6)

    def test_qcc_H2(self):
        """ Verify closed-shell QCC functionalities for H2 """

        # Build the QCC class for H2, which sets the QMF parameters automatically if None are passed
        qcc_op_list = [QubitOperator("X0 Y1 Y2 Y3")]
        qcc_var_params = [0.2261362655507856]
        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True, qubit_op_list=qcc_op_list)

        # Build QMF and QCC circuits, prepending the former to the latter 
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701746609022, delta=1e-6)

    def test_qmf_qcc_H4(self):
        """ Verify closed-shell QMF + QCC functionalities for H4 """

        # Determine the QMF parameter set and cicuit for H4
        qmf_ansatz = QMF(mol_H4_sto3g, up_then_down=True)
        qmf_ansatz.build_circuit()

        # Build the QCC class and pass the QMF parameters and circuit
        qcc_op_list = [QubitOperator("Y1 X3 Y5 Y7"), QubitOperator("Y0 Y2 Y4 X6"), QubitOperator("X1 X2 Y5 X6"), QubitOperator("X0 X2 X5 Y7"),
                       QubitOperator("X1 X3 X4 Y6"), QubitOperator("X0 X3 Y4 X7"), QubitOperator("Y0 Y2 Y5 X6"), QubitOperator("Y1 Y2 X4 Y6"),
                       QubitOperator("X0 X3 X5 Y6"), QubitOperator("X1 X2 X4 Y7")]
        qcc_var_params = [ 1.16537736e-02,  2.36253847e-02, -1.35574773e+00,  2.95424730e-01,
                           2.97982293e-01, -1.85994445e-01,  7.70410338e-02, -7.81960266e-02,
                           1.86621357e-02,  1.57873932e-01]
        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list, qmf_var_params=qmf_ansatz.var_params, qmf_circuit=qmf_ansatz.circuit)

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9515395411986798, delta=1e-6)

    def test_qcc_H4(self):
        """ Verify closed-shell QCC functionalities for H4 """

        # Build the QCC class for H4, which sets the QMF parameters automatically if None are passed
        qcc_op_list = [QubitOperator("Y1 X3 Y5 Y7"), QubitOperator("Y0 Y2 Y4 X6"), QubitOperator("X1 X2 Y5 X6"), QubitOperator("X0 X2 X5 Y7"),
                       QubitOperator("X1 X3 X4 Y6"), QubitOperator("X0 X3 Y4 X7"), QubitOperator("Y0 Y2 Y5 X6"), QubitOperator("Y1 Y2 X4 Y6"),
                       QubitOperator("X0 X3 X5 Y6"), QubitOperator("X1 X2 X4 Y7")]
        qcc_var_params = [ 1.16537736e-02,  2.36253847e-02, -1.35574773e+00,  2.95424730e-01,
                           2.97982293e-01, -1.85994445e-01,  7.70410338e-02, -7.81960266e-02,
                           1.86621357e-02,  1.57873932e-01]
        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list) 

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9515395411986798, delta=1e-6)

if __name__ == "__main__":
    unittest.main()
