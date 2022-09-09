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

"""Unit tests for closed-shell and restricted open-shell qubit coupled cluster
with involutory linear combinations (ILC) of anticommuting sets (ACS) of Pauli words."""

import unittest

import numpy as np

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.ilc import ILC
from tangelo.toolboxes.ansatz_generator._qubit_ilc import gauss_elim_over_gf2
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_cation_sto3g

sim = Simulator()


class ILCTest(unittest.TestCase):
    """Unit tests for various functionalities of the ILC ansatz class. Examples for both closed-
    and restricted open-shell ILC are provided using H2, H4, and H4+.
    """

    @staticmethod
    def test_ilc_set_var_params():
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        ilc_ansatz = ILC(mol_H2_sto3g, up_then_down=True)

        nine_zeros = np.zeros((9,), dtype=float)

        ilc_ansatz.set_var_params([0.] * 9)
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, nine_zeros, decimal=6)

        nine_tenths = 0.1 * np.ones((9,))

        ilc_ansatz.set_var_params([0.1] * 9)
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, nine_tenths, decimal=6)

        ilc_ansatz.set_var_params(np.array([0.1] * 9))
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, nine_tenths, decimal=6)

    def test_ilc_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        ilc_ansatz = ILC(mol_H2_sto3g, up_then_down=True)

        self.assertRaises(ValueError, ilc_ansatz.set_var_params, np.array([1.] * 2))

    @staticmethod
    def test_gauss_elim_over_gf2_sqrmat():
        """ Verify behavior of the Gaussian elimination for a square matrix."""

        # a_matrix stores the action of A * z over GF(2); dimension is n x m
        a_matrix = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0], [0, 0, 1, 0]])

        # b_vec stores the solution vector for the equation A * z = b_vec; dimension is n x 1
        b_vec = np.array([1, 0, 1, 0]).reshape((4, 1))

        # z_ref stores the serves as the reference for the output of gauss_elim_over_gf2
        z_ref = np.array([0, 1, 0, 1])

        # solve A * z = b and compare to reference solution
        z_sln = gauss_elim_over_gf2(a_matrix, b_vec)

        np.testing.assert_array_almost_equal(z_sln, z_ref, decimal=6)

    @staticmethod
    def test_gauss_elim_over_gf2_rectmat():
        """ Verify behavior of the Gaussian elimination for a rectangular matrix."""

        # a_matrix stores the action of A * z over GF(2); dimension is n x m
        a_matrix = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]])

        # b_vec stores the solution vector for the equation A * z = b_vec; dimension is n x 1
        b_vec = np.array([1, 1, 0]).reshape((3, 1))

        # z_ref stores the serves as the reference for the output of gauss_elim_over_gf2
        z_ref = np.array([1, 0, 1, 0, 0])

        # solve A * z = b and compare to reference solution
        z_sln = gauss_elim_over_gf2(a_matrix, b_vec)

        np.testing.assert_array_almost_equal(z_sln, z_ref, decimal=6)

    @staticmethod
    def test_gauss_elim_over_gf2_lindep():
        """ Verify behavior of the Gaussian elimination when linear dependence
        in the form of duplicate rows arises in a matrix."""

        # a_matrix stores the action of A * z over GF(2); dimension is n x m
        a_matrix = np.array([[0, 0, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1]])

        # b_vec stores the solution vector for the equation A * z = b_vec; dimension is n x 1
        b_vec = np.array([0, 0, 1]).reshape((3, 1))

        # z_ref stores the serves as the reference for the output of gauss_elim_over_gf2
        z_ref = np.array([0, 0, 0, 1, 0])

        # solve A * z = b and compare to reference solution
        z_sln = gauss_elim_over_gf2(a_matrix, b_vec)

        np.testing.assert_array_almost_equal(z_sln, z_ref, decimal=6)

    @staticmethod
    def test_gauss_elim_over_gf2_lindep2():
        """ Verify behavior of the Gaussian elimination when linear dependence
        in the form of a row of zeros arises in a matrix."""

        # a_matrix stores the action of A * z over GF(2); dimension is n x m
        a_matrix = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]])

        # b_vec stores the solution vector for the equation A * z = b_vec; dimension is n x 1
        b_vec = np.array([1, 1, 0]).reshape((3, 1))

        # z_ref stores the serves as the reference for the output of gauss_elim_over_gf2
        z_ref = np.array([1, 1, 0, 0, 0])

        # solve A * z = b and compare to reference solution
        z_sln = gauss_elim_over_gf2(a_matrix, b_vec)

        np.testing.assert_array_almost_equal(z_sln, z_ref, decimal=6)

    def test_ilc_h2(self):
        """ Verify closed-shell functionality when using the ILC class separately for H2."""

        # Specify the qubit operators from the anticommuting set (ACS) of ILC generators.
        acs = [QubitOperator("Y0 X1")]
        ilc_ansatz = ILC(mol_H2_sto3g, mapping="scbk", up_then_down=True, acs=acs)

        # Build the ILC circuit, which is prepended by the qubit mean field (QMF) circuit.
        ilc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = ilc_ansatz.qubit_ham

        # The QMF and ILC parameters can both be specified; determined automatically otherwise.
        qmf_var_params = [3.14159265e+00,  3.14159265e+00, -7.59061327e-12,  0.]
        ilc_var_params = [1.12894599e-01]
        var_params = qmf_var_params + ilc_var_params
        ilc_ansatz.update_var_params(var_params)
        # Assert energy returned is as expected for given parameters
        energy = sim.get_expectation_value(qubit_hamiltonian, ilc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.137270126, delta=1e-6)

    def test_ilc_h4_cation(self):
        """ Verify restricted open-shell functionality when using the ILC class for H4+ """

        # Specify the qubit operators from the anticommuting set (ACS) of ILC generators.
        acs = [QubitOperator("X0 X1 X2 X3 X4 Y5"), QubitOperator("X1 Z2 X3 X4 Y5"),
               QubitOperator("Y0 X1 Z2 X3 X4 Z5"), QubitOperator("Z0 X1 X2 X3 X4 Y5"),
               QubitOperator("X1 Y2 X3 X4"), QubitOperator("Y1 X3 X4"),
               QubitOperator("Y0 X1 Z2 X3 X4 X5"), QubitOperator("Y0 X1 X2 X3 X4")]
        ilc_ansatz = ILC(mol_H4_cation_sto3g, mapping="scbk", up_then_down=True, acs=acs)

        # Build the ILC circuit, which is prepended by the qubit mean field (QMF) circuit.
        ilc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = ilc_ansatz.qubit_ham

        # The QMF and ILC parameters can both be specified; determined automatically otherwise.
        qmf_var_params = [ 3.14159265e+00, -1.02576971e-11,  1.35522331e-11,  3.14159265e+00,
                           3.14159265e+00, -5.62116001e-11, -1.41419277e-11, -2.36789365e-11,
                          -5.53225030e-11, -3.56400157e-11, -2.61030058e-11, -3.55652002e-11]
        ilc_var_params = [ 0.14001419, -0.10827113,  0.05840200, -0.12364925,
                          -0.07275071, -0.04703495,  0.02925292,  0.03145765]
        var_params = qmf_var_params + ilc_var_params
        # Assert energy returned is as expected for given parameters
        ilc_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, ilc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.6379638, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
