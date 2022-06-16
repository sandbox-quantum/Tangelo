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

"""Unit tests for closed-shell and restricted open-shell qubit coupled cluster (QCC) ansatze. """

import os
import unittest

import numpy as np
from openfermion import load_operator

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_cation_sto3g, mol_H4_doublecation_minao

sim = Simulator()

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class QCCTest(unittest.TestCase):
    """Unit tests for various functionalities of the QCC ansatz class. Examples for both closed-
    and restricted open-shell QCC are provided using H2, H4+, and H4+2."""

    @staticmethod
    def test_qcc_set_var_params():
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True)

        nine_zeros = np.zeros((9,), dtype=float)

        qcc_ansatz.set_var_params([0.] * 9)
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, nine_zeros, decimal=6)

        nine_tenths = 0.1 * np.ones((9,))

        qcc_ansatz.set_var_params([0.1] * 9)
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, nine_tenths, decimal=6)

        qcc_ansatz.set_var_params(np.array([0.1] * 9))
        np.testing.assert_array_almost_equal(qcc_ansatz.var_params, nine_tenths, decimal=6)

    def test_qcc_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True)

        self.assertRaises(ValueError, qcc_ansatz.set_var_params, np.array([1.] * 2))

    def test_qcc_h2(self):
        """ Verify closed-shell functionality when using the QCC class for H2 """

        # Specify the qubit operators from the direct interaction set (DIS) of QCC generators.
        dis = [QubitOperator("Y0 X1 X2 X3")]
        qcc_ansatz = QCC(mol_H2_sto3g, up_then_down=True, dis=dis)

        # Build the QCC circuit, which is prepended by the qubit mean field (QMF) circuit.
        qcc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # The QMF and QCC parameters can both be specified; determined automatically othersise.
        qmf_var_params = [ 3.14159265e+00, -2.42743256e-08,  3.14159266e+00, -3.27162543e-08,
                           3.08514545e-09,  3.08514545e-09,  3.08514545e-09,  3.08514545e-09]
        qcc_var_params = [-2.26136280e-01]
        var_params = qmf_var_params + qcc_var_params
        # Assert energy returned is as expected for given parameters
        qcc_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701746609022, delta=1e-6)

    def test_qmf_qcc_h4_cation(self):
        """ Verify restricted open-shell functionality when using QCC ansatz for H4+ """

        # Specify the qubit operators from the direct interaction set (DIS) of QCC generators.
        dis = [QubitOperator("X0 X1 Y2 Y3 X4 Y5"), QubitOperator("Y1 Y3 Y4 X5"),
               QubitOperator("X0 Y1 Y3 Y4"), QubitOperator("X1 X2 Y3 X4 X5"),
               QubitOperator("Y1 Y2 Y3 X4"), QubitOperator("Y1 X3 X4"),
               QubitOperator("Y0 X2"), QubitOperator("X0 X1 X3 X4 Y5"),
               QubitOperator("X0 X1 X2 Y3 X4")]
        qcc_ansatz = QCC(mol_H4_cation_sto3g, "scbk", True, dis)

        # Build the QCC circuit, which is prepended by the qubit mean field (QMF) circuit.
        qcc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qcc_ansatz.qubit_ham

        # The QMF and QCC parameters can both be specified; determined automatically othersise.
        qmf_var_params = [3.14159302e+00,  6.20193478e-07,  1.51226426e-06,  3.14159350e+00,
                          3.14159349e+00,  7.88310582e-07,  3.96032530e+00,  2.26734374e+00,
                          3.22127001e+00,  5.77997401e-01,  5.51422406e+00,  6.26513711e+00]
        qcc_var_params = [-0.26816042,  0.21694796,  0.12139543, -0.2293093,
                          -0.14577423, -0.08937818,  0.01796464, -0.06445363,  0.06056016]
        var_params = qmf_var_params + qcc_var_params
        # Assert energy returned is as expected for given parameters
        qcc_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.6382913, delta=1e-6)

    def test_qmf_qcc_h4_double_cation(self):
        """ Verify restricted open-shell functionality when using the QCC ansatz for H4+2 """

        # Specify the qubit operators from the direct interaction set (DIS) of QCC generators.
        dis = [QubitOperator("Y0 X4"), QubitOperator("X0 Y1 Y2 X4 Y5 X6"),
               QubitOperator("Y0 Y1 Y4 X5"), QubitOperator("Y0 X1 Y4 Y5 X6"),
               QubitOperator("X0 X1 X2 Y4 X5")]
        qcc_ansatz = QCC(mol_H4_doublecation_minao, "bk", True, dis)

        # Build the QCC circuit, which is prepended by the qubit mean field (QMF) circuit.
        qcc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = load_operator("mol_H4_doublecation_minao_qubitham_bk_updown.data", data_directory=pwd_this_test+"/data", plain_text=True)

        # The QMF and QCC parameters can both be specified; determined automatically othersise.
        qmf_var_params = [3.14159247e+00,  3.14158884e+00,  1.37660700e-06,  3.14159264e+00,
                          3.14159219e+00,  3.14158908e+00,  0.00000000e+00,  0.00000000e+00,
                          6.94108155e-01,  1.03928030e-01,  5.14029803e+00,  2.81850365e+00,
                          4.25403875e+00,  6.19640367e+00,  1.43241026e+00,  3.50279759e+00]
        qcc_var_params = [-2.76489925e-01, -2.52783324e-01,  5.76565629e-02,  6.99988237e-02,
                          -7.03721438e-02]
        var_params = qmf_var_params + qcc_var_params
        # Assert energy returned is as expected for given parameters
        qcc_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qcc_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.85465810, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
