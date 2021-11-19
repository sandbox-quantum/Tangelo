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

        # Build the QMF ansatz with optimized variational parameters
        qmf_var_params = [ 3.14159265e+00, -2.42743256e-08,  3.14159266e+00, -3.27162543e-08,
                           3.08514545e-09,  3.08514545e-09,  3.08514545e-09,  3.08514545e-09 ]
        qmf_ansatz = QMF(mol_H2_sto3g, up_then_down=True)
        qmf_ansatz.build_circuit(qmf_var_params)

        # Build the QCC ansatz with a QCC generator and optimized parameter; pass the optimized QMF parameters and circuit
        qcc_op_list = [QubitOperator("Y0 X1 X2 X3")]
        qcc_var_params = [-2.26136280e-01]
        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True, qubit_op_list=qcc_op_list, qmf_var_params=qmf_ansatz.var_params, qmf_circuit=qmf_ansatz.circuit)

        # Build a variational QCC circuit and prepend a variational QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham 

        # Assert energy returned is as expected for the optimized QMF + QCC variational parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)
                                      
        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.137270174660901, delta=1e-6)

    def test_qcc_H2(self):
        """ Verify closed-shell QCC functionalities for H2 """

        # Build the QCC class for H2, which sets the QMF parameters automatically if None are passed
        qcc_op_list = [QubitOperator("X0 Y1 Y2 Y3")]
        qcc_var_params = [0.22613627]
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

        # Build the QMF ansatz with optimized variational parameters
        qmf_var_params = [ 3.14159240e+00,  3.14159212e+00, -5.23016857e-07,  1.44620161e-07,
                           3.14159270e+00,  3.14159273e+00, -3.68840303e-07,  6.71472866e-08,
                          -8.70574300e-08, -8.70574300e-08,  4.49523312e-07,  2.33914807e-07,
                           2.14488065e-06,  1.11626300e-06,  2.20341542e-06,  6.62639874e-07 ]
        qmf_ansatz = QMF(mol_H4_sto3g, up_then_down=True)
        qmf_ansatz.build_circuit(qmf_var_params)

        # Build the QCC ansatz with a QCC generator and optimized parameter; pass the optimized QMF parameters and circuit
        qcc_op_list = [QubitOperator("Y1 Y3 X5 Y7"), QubitOperator("Y0 Y2 X4 Y6"), QubitOperator("Y1 X2 X5 X6"), QubitOperator("X0 Y2 X5 X7"),
                       QubitOperator("Y1 Y3 Y4 X6"), QubitOperator("Y0 Y3 Y4 X7"), QubitOperator("X0 X2 X5 Y6"), QubitOperator("Y1 Y2 Y4 X6"),
                       QubitOperator("Y0 X3 Y5 Y6"), QubitOperator("X1 Y2 Y4 Y7"), QubitOperator("Y0 Y3 X5 Y7"), QubitOperator("X1 Y3 Y4 Y7"),
                       QubitOperator("X0 X1 Y2 X3"), QubitOperator("X4 Y5 Y6 Y7"), QubitOperator("X1 Y3 Y5 Y6"), QubitOperator("X1 Y2 Y5 Y7"),
                       QubitOperator("X0 Y3 Y4 Y6"), QubitOperator("Y0 X2 Y4 Y7")]
        qcc_var_params = [  8.13594173e-02,  5.17649672e-02, -1.31132560e+00,  2.56564526e-01,
                           -2.31674482e-01, -1.30875865e-01, -7.89718913e-02,  8.01799962e-02,
                           -2.25464402e-01,  3.64317733e-02,  3.92029443e-02,  4.27843829e-02,
                           -2.48747025e-03, -2.10290191e-01,  3.93801396e-02, -8.45929757e-03,
                            2.48311883e-02, -3.53866908e-02 ]
        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list, qmf_var_params=qmf_ansatz.var_params, qmf_circuit=qmf_ansatz.circuit)

        # Build a variational QCC circuit and prepend a variational QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for the optimized QMF + QCC variational parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9659086296992885, delta=1e-6)

    def test_qcc_H4(self):
        """ Verify closed-shell QCC functionalities for H4 """

        # Build the QCC class for H4, which sets the QMF parameters automatically if None are passed
        qcc_op_list = [QubitOperator("Y1 Y3 X5 Y7"), QubitOperator("Y0 Y2 X4 Y6"), QubitOperator("Y1 X2 X5 X6"), QubitOperator("X0 Y2 X5 X7"),
                       QubitOperator("Y1 Y3 Y4 X6"), QubitOperator("Y0 Y3 Y4 X7"), QubitOperator("X0 X2 X5 Y6"), QubitOperator("Y1 Y2 Y4 X6"),
                       QubitOperator("Y0 X3 Y5 Y6"), QubitOperator("X1 Y2 Y4 Y7"), QubitOperator("Y0 Y3 X5 Y7"), QubitOperator("X1 Y3 Y4 Y7"),
                       QubitOperator("X0 X1 Y2 X3"), QubitOperator("X4 Y5 Y6 Y7"), QubitOperator("X1 Y3 Y5 Y6"), QubitOperator("X1 Y2 Y5 Y7"),
                       QubitOperator("X0 Y3 Y4 Y6"), QubitOperator("Y0 X2 Y4 Y7")]
        qcc_var_params = [ 0.08134942,  0.05176103, -1.31130635,  0.25658127,
                          -0.23169954, -0.13088206, -0.07897360,  0.08017313, 
                          -0.22544533,  0.03642924,  0.03918970,  0.04276501,
                          -0.00248156, -0.21026984,  0.03939769, -0.00847347,
                           0.02479605, -0.03540217 ]
        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list) 

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.965908629987424, delta=1e-6)

if __name__ == "__main__":
    unittest.main()
