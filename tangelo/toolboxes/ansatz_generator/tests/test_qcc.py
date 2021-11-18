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
        qcc_op_list = [QubitOperator("Y1 Y3 X5 Y7"), QubitOperator("X0 Y2 X4 X6"), QubitOperator("Y1 Y2 X5 Y6"), QubitOperator("Y0 X2 X5 X7"),
                       QubitOperator("Y1 X3 X4 X6"), QubitOperator("X0 Y3 X4 X7"), QubitOperator("Y0 Y2 Y5 X6"), QubitOperator("X1 Y2 X4 X6"),
                       QubitOperator("Y0 X3 Y5 Y6"), QubitOperator("Y1 Y2 X4 Y7"), QubitOperator("X0 Y3 X5 X7"), QubitOperator("X1 X3 X4 Y7"),
                       QubitOperator("Y0 X1 Y2 Y3"), QubitOperator("X4 X5 X6 Y7"), QubitOperator("X1 Y3 Y5 Y6"), QubitOperator("X1 X2 X5 Y7"),
                       QubitOperator("X0 Y3 X4 X6"), QubitOperator("Y0 Y2 X4 Y7")]

        qcc_var_params = [ 1.42887596e-02,  3.56947444e-02,  1.32707789e+00, -2.52155088e-01,
                          -2.52985326e-01,  1.75199217e-01,  7.90798760e-02, -8.73381559e-02,
                          -2.48431517e-01,  1.81922659e-01,  3.35759294e-02,  2.72440813e-02,
                          -4.50642480e-03, -6.39186375e-02,  1.31620822e-03, -5.42781311e-03,
                           1.09710651e-02,  3.39231314e-02 ]

        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list, qmf_var_params=qmf_ansatz.var_params, qmf_circuit=qmf_ansatz.circuit)

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9687959852258365, delta=1e-6)

    def test_qcc_H4(self):
        """ Verify closed-shell QCC functionalities for H4 """

        # Build the QCC class for H4, which sets the QMF parameters automatically if None are passed
        qcc_op_list = [QubitOperator("Y1 Y3 X5 Y7"), QubitOperator("X0 Y2 X4 X6"), QubitOperator("Y1 Y2 X5 Y6"), QubitOperator("Y0 X2 X5 X7"),
                       QubitOperator("Y1 X3 X4 X6"), QubitOperator("X0 Y3 X4 X7"), QubitOperator("Y0 Y2 Y5 X6"), QubitOperator("X1 Y2 X4 X6"),
                       QubitOperator("Y0 X3 Y5 Y6"), QubitOperator("Y1 Y2 X4 Y7"), QubitOperator("X0 Y3 X5 X7"), QubitOperator("X1 X3 X4 Y7"),
                       QubitOperator("Y0 X1 Y2 Y3"), QubitOperator("X4 X5 X6 Y7"), QubitOperator("X1 Y3 Y5 Y6"), QubitOperator("X1 X2 X5 Y7"),
                       QubitOperator("X0 Y3 X4 X6"), QubitOperator("Y0 Y2 X4 Y7")]

        qcc_var_params = [ 1.42887596e-02,  3.56947444e-02,  1.32707789e+00, -2.52155088e-01,
                          -2.52985326e-01,  1.75199217e-01,  7.90798760e-02, -8.73381559e-02,
                          -2.48431517e-01,  1.81922659e-01,  3.35759294e-02,  2.72440813e-02,
                          -4.50642480e-03, -6.39186375e-02,  1.31620822e-03, -5.42781311e-03,
                           1.09710651e-02,  3.39231314e-02 ]
 
        qcc_ansatz = QCC(mol_H4_sto3g, up_then_down=True, qubit_op_list=qcc_op_list) 

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9687959852258365, delta=1e-6)

if __name__ == "__main__":
    unittest.main()
