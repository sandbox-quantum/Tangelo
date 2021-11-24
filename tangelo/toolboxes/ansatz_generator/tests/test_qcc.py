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
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_cation_sto3g, mol_H4_doublecation_minao

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

    def test_qcc_H4_cation(self):
        """ Verify open-shell QMF + QCC functionalities for H4 +1 """

        # Build the QCC class for H4 1+ with QMF parameters and optimized QCC parameters for a set of generators.
        qmf_var_params = [ 3.14159265, 0.,         0.,         3.14159265, 3.14159265, 0.,
                           1.65188077, 0.34371929, 4.43025577, 4.26598251, 1.08692256, 1.29529587 ]

        qcc_op_list = [ QubitOperator("X0 X1 Y2 X3 X4 X5"), QubitOperator("X1 Y3 Y4 Y5"), QubitOperator("Y0 Y1 X3 Y4"), QubitOperator("X1 X2 Y3 Y4 Y5"),
                        QubitOperator("X1 X2 X3 Y4"),       QubitOperator("X1 Y3 X4"),    QubitOperator("Y0 X2"),       QubitOperator("X0 Y1 Y3 Y4 X5"),
                        QubitOperator("Y0 Y1 X2 X3 Y4") ]

        qcc_var_params = [ -0.26197751,   0.2181322,  0.11635916, -0.22256682,
                            0.14391785,  0.08627913,  0.03102110,  0.06512990,
                            0.05254937 ]

        qcc_ansatz = QCC(mol_H4_cation_sto3g, mapping="SCBK", qubit_op_list=qcc_op_list, qmf_var_params=qmf_var_params)

        # Build a variational QCC circuit and prepend a variational QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for the optimized QMF + QCC variational parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.63781775, delta=1e-6)

    def test_qcc_H4_double_cation(self):
        """ Verify closed-shell QCC functionalities for H4 +2 """

        # Build the QCC class for H4 2+ with QMF parameters and optimized QCC parameters for a set of generators.
        qmf_var_params = [ 3.14159265, 0.,         0.,         0.,         0.,         0.,
                           0.,         0.,         2.68274492, 0.65222886, 1.61056342, 5.41758547,
                           3.92606034, 4.05125009, 0.03740878, 2.86163151 ]

        qcc_op_list = [ QubitOperator("X0 Y2"), QubitOperator("Y0 X4"), QubitOperator("X0 Y6"), QubitOperator("X0 Y5 X6"),
                        QubitOperator("Y0 Y4 Y5") ]                   
                      
        qcc_var_params = [ 0.27697283, -0.2531527,  0.05947973, -0.06943673,
                           0.07049098 ]
                       
        qcc_ansatz = QCC(mol_H4_doublecation_minao, mapping="BK", qmf_var_params=qmf_var_params, qubit_op_list=qcc_op_list) 

        # Build the QCC circuit and prepend the QMF circuit to it
        qcc_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        qcc_ansatz.update_var_params(qcc_var_params)

        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.8547019, delta=1e-6)

if __name__ == "__main__":
    unittest.main()
