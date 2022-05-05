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

"""Unit tests for closed-shell and restricted open-shell qubit mean field (QMF) ansatz. """

import unittest

import numpy as np

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g

sim = Simulator()


class QMFTest(unittest.TestCase):
    """Unit tests of various functionalities of the QMF ansatz class. Examples for both closed-
    and restricted open-shell QMF are provided using H2, H4, and H4+.
    """

    @staticmethod
    def test_qmf_set_var_params():
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        qmf_ansatz = QMF(mol_H2_sto3g)

        eight_zeros = np.zeros((8,), dtype=float)

        qmf_ansatz.set_var_params("vacuum")
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_zeros, decimal=6)

        qmf_ansatz.set_var_params([0.] * 8)
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_zeros, decimal=6)

        eight_pis = np.pi * np.ones((8,))

        qmf_ansatz.set_var_params("full_pi")
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_pis, decimal=6)

        qmf_ansatz.set_var_params(np.array([np.pi] * 8))
        np.testing.assert_array_almost_equal(qmf_ansatz.var_params, eight_pis, decimal=6)

    def test_qmf_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)

        self.assertRaises(ValueError, qmf_ansatz.set_var_params, np.array([1.] * 4))

    @staticmethod
    def test_qmf_set_params_upper_hf_state_h2():
        """ Verify closed-shell QMF functionalities for H2: upper case initial parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("HF_State")

        expected = np.array([np.pi] * 2 + [0.] * 6)
        np.testing.assert_allclose(expected, qmf_ansatz.var_params, rtol=1e-10)

    @staticmethod
    def test_qmf_set_params_lower_hf_state_h2():
        """ Verify closed-shell QMF functionalities for H2: lower case initial parameters """

        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.set_var_params("hf_state")

        expected = np.array([np.pi] * 2 + [0.] * 6)
        np.testing.assert_allclose(expected, qmf_ansatz.var_params, rtol=1e-10)

    @staticmethod
    def test_qmf_set_params_hf_state_h4():
        """ Verify closed-shell QMF functionalities for H4: hf_state initial parameters """

        qmf_ansatz = QMF(mol_H4_sto3g)
        qmf_ansatz.set_var_params("hf_state")

        expected = np.array([np.pi] * 4 + [0.] * 12)
        np.testing.assert_allclose(expected, qmf_ansatz.var_params, rtol=1e-10)

    def test_qmf_closed_h2(self):
        """ Verify closed-shell QMF functionalities for H2 """

        # Build ansatz and circuit
        qmf_var_params = [np.pi] * 2 + [0.] * 6
        qmf_ansatz = QMF(mol_H2_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        qmf_ansatz.update_var_params(qmf_var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1166843870853400, delta=1e-6)

    def test_qmf_closed_h4(self):
        """ Verify closed-shell QMF functionalities for H4. """

        # Build ansatz and circuit
        qmf_var_params = [np.pi] * 4 + [0.] * 12
        qmf_ansatz = QMF(mol_H4_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        qmf_ansatz.update_var_params(qmf_var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.7894832518559973, delta=1e-6)

    def test_qmf_open_h4_cation(self):
        """ Verify open-shell QMF functionalities for H4 + """

        # Build ansatz and circuit
        qmf_var_params = [np.pi] * 4 + [0.] * 12
        qmf_ansatz = QMF(mol_H4_cation_sto3g)
        qmf_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = qmf_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        energy = sim.get_expectation_value(qubit_hamiltonian, qmf_ansatz.circuit)
        qmf_ansatz.update_var_params(qmf_var_params)
        self.assertAlmostEqual(energy, -1.5859184313544759, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
