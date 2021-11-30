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

"""Unit tests for closed-shell and restricted open-shell qubit mean field (QMF) ansatze. """

import unittest
import numpy as np

from tangelo.backendbuddy import Simulator
from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g

sim = Simulator(target="qulacs")


class QMFTest(unittest.TestCase):
    """Unit tests of various functionalities of the QMF ansatz class. Examples for both closed-
    and restricted open-shell QMF are provided using H2, H4, and H4+.
    """

    @staticmethod
    def test_qmf_set_var_params():
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

    def test_qmf_set_params_upper_hf_state_h2(self):
        """ Verify closed-shell QMF functionalities for H2: upper case initial parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("HF-State")

        # only check the theta Bloch angles -- phi Bloch angles are randomly initialized
        qmf_thetas = qmf_ansatz.var_params[:4]
        expected = [3.141592653589793, 3.141592653589793, 0., 0.]

        self.assertAlmostEqual(np.linalg.norm(qmf_thetas), np.linalg.norm(expected), delta=1e-10)

    def test_qmf_set_params_lower_hf_state_h2(self):
        """ Verify closed-shell QMF functionalities for H2: lower case initial parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("hf-state")

        # only check the theta Bloch angles -- phi Bloch angles are randomly initialized
        qmf_thetas = qmf_ansatz.var_params[:4]
        expected = [3.141592653589793, 3.141592653589793, 0., 0.]

        self.assertAlmostEqual(np.linalg.norm(qmf_thetas), np.linalg.norm(expected), delta=1e-10)

    def test_qmf_set_params_hf_state_h4(self):
        """ Verify closed-shell QMF functionalities for H4: hf-state initial parameters """

        qmf_ansatz = QMF(mol_H4_sto3g)
        qmf_ansatz.set_var_params("hf-state")

        # only check the theta Bloch angles -- phi Bloch angles are randomly initialized
        qmf_thetas = qmf_ansatz.var_params[:8]
        expected = [3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793,
                    0.,                0.,                0.,                0.]

        self.assertAlmostEqual(np.linalg.norm(qmf_thetas), np.linalg.norm(expected), delta=1e-10)

    def test_qmf_closed_h2(self):
        """ Verify closed-shell QMF functionalities for H2 """

        var_params = [3.14159265, 3.14159265, 0.,         0.,
                      4.61265659, 0.73017920, 1.03851163, 2.48977533]
        # Build circuit
        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        qmf_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1166843870853400, delta=1e-6)

    def test_qmf_closed_h4(self):
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
        qmf_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.7894832518559973, delta=1e-6)

    def test_qmf_open_h4_cation(self):
        """ Verify open-shell QMF functionalities for H4 + """

        var_params = [3.14159265, 3.14159265, 3.14159265, 0.,
                      0.,         0.,         0.,         0.,
                      2.56400050, 5.34585441, 1.46689000, 1.3119943,
                      2.95766833, 5.00079708, 3.53150391, 1.9093635]

        # Build circuit
        qmf_ansatz = QMF(mol_H4_cation_sto3g, qmf_state_init="hf-state")
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        qmf_ansatz.update_var_params(var_params)
        self.assertAlmostEqual(energy, -1.5859184313544759, delta=1e-6)


if __name__ == "__main__":
    unittest.main()

