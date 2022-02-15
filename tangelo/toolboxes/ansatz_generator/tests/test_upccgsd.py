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
import os
from openfermion import load_operator

from tangelo.molecule_library import mol_H2_sto3g, mol_H4_doublecation_minao, mol_H4_cation_sto3g
from tangelo.toolboxes.qubit_mappings import jordan_wigner
from tangelo.toolboxes.ansatz_generator.upccgsd import UpCCGSD
from tangelo.linq import Simulator

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class UpCCGSDTest(unittest.TestCase):

    def test_upccgsd_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (keyword,
        list, numpy array).
        """

        upccgsd_ansatz = UpCCGSD(mol_H2_sto3g)

        six_ones = np.ones((6,))

        upccgsd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, six_ones, decimal=6)

        upccgsd_ansatz.set_var_params([1., 1., 1., 1., 1., 1.])
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, six_ones, decimal=6)

        upccgsd_ansatz.set_var_params(np.array([1., 1., 1., 1., 1., 1.]))
        np.testing.assert_array_almost_equal(upccgsd_ansatz.var_params, six_ones, decimal=6)

    def test_upccgsd_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        upccgsd_ansatz = UpCCGSD(mol_H2_sto3g)

        self.assertRaises(ValueError, upccgsd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_upccgsd_H2(self):
        """Verify closed-shell UpCCGSD functionalities for H2."""

        # Build circuit
        upccgsd_ansatz = UpCCGSD(mol_H2_sto3g)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        upccgsd_ansatz.update_var_params([0.03518165, -0.02986551,  0.02897598, -0.03632711,
                                          0.03044071,  0.08252277])
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372658, delta=1e-6)

    def test_upccgsd_H4_open(self):
        """Verify open-shell UpCCGSD functionalities for H4."""

        var_params = [-0.0291763,   0.36927821,  0.14654907, -0.13845063,  0.14387348, -0.00903457,
                      -0.56843484,  0.01223853,  0.13649942,  0.83225887,  0.20236275,  0.02682977,
                      -0.17198068,  0.10161518,  0.01523924,  0.30848876,  0.22430705, -0.07290468,
                       0.16253591,  0.02268874,  0.2382988,   0.33716289, -0.20094664,  0.3057071,
                      -0.58426117,  0.22433297,  0.29668267,  0.64761217, -0.2705204,   0.07540534,
                       0.20131878,  0.09890588,  0.10563459, -0.22983007,  0.13578206, -0.02017009]

        # Build circuit
        upccgsd_ansatz = UpCCGSD(mol_H4_cation_sto3g)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H4_cation_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        upccgsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.6412047312, delta=1e-6)

    def test_upccgsd_H4_doublecation(self):
        """Verify closed-shell UpCCGSD functionalities for H4 2+."""

        var_params = [1.08956248, 1.08956247, 1.05305993, 1.05305993, 0.8799399, 0.8799399,
                      0.88616586, 0.88616586, 1.09532143, 1.09532143, 1.23586857, 1.23586857,
                      1.09001216, 0.85772769, 1.28020861, 1.05820721, 0.9680792,  1.01693601,
                      0.68355852, 0.68355852, 1.30303827, 1.30303827, 0.74524063, 0.74524063,
                      0.36958813, 0.36958813, 1.37092805, 1.37092805, 0.92860293, 0.92860293,
                      1.30296676, 0.5803438,  1.42469953, 1.05666723, 0.86961358, 0.55347531]

        # Build circuit
        upccgsd_ansatz = UpCCGSD(mol_H4_doublecation_minao)
        upccgsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = load_operator("mol_H4_doublecation_minao_qubitham_jw.data", data_directory=pwd_this_test+"/data", plain_text=True)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        upccgsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, upccgsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.854608, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
