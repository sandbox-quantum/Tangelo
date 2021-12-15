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

from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_doublecation_minao, mol_H4_cation_sto3g
from tangelo.toolboxes.qubit_mappings import jordan_wigner
from tangelo.toolboxes.ansatz_generator.uccsd import UCCSD
from tangelo.linq import Simulator

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class UCCSDTest(unittest.TestCase):

    def test_uccsd_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (keyword, list, numpy array).
        MP2 have their own tests.
        """

        uccsd_ansatz = UCCSD(mol_H2_sto3g)

        two_ones = np.ones((2,))

        uccsd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, two_ones, decimal=6)

        uccsd_ansatz.set_var_params([1., 1.])
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, two_ones, decimal=6)

        uccsd_ansatz.set_var_params(np.array([1., 1.]))
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, two_ones, decimal=6)

    def test_uccsd_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        uccsd_ansatz = UCCSD(mol_H2_sto3g)

        self.assertRaises(ValueError, uccsd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_uccsd_set_params_MP2_H2(self):
        """Verify closed-shell UCCSD functionalities for H2: MP2 initial
        parameters.
        """

        uccsd_ansatz = UCCSD(mol_H2_sto3g)
        uccsd_ansatz.set_var_params("MP2")

        expected = [2e-05, 0.0363253711023451]
        self.assertAlmostEqual(np.linalg.norm(uccsd_ansatz.var_params), np.linalg.norm(expected), delta=1e-10)

    def test_uccsd_set_params_mp2_H2(self):
        """Verify closed-shell UCCSD functionalities for H2: lower case mp2
        initial parameters.
        """

        uccsd_ansatz = UCCSD(mol_H2_sto3g)
        uccsd_ansatz.set_var_params("mp2")

        expected = [2e-05, 0.0363253711023451]
        self.assertAlmostEqual(np.linalg.norm(uccsd_ansatz.var_params), np.linalg.norm(expected), delta=1e-10)

    def test_uccsd_set_params_MP2_H4(self):
        """Verify closed-shell UCCSD functionalities for H4: MP2 initial parameters """

        uccsd_ansatz = UCCSD(mol_H4_sto3g)
        uccsd_ansatz.set_var_params("MP2")

        expected = [2e-05, 2e-05, 2e-05, 2e-05, 0.03894901872789466, 0.07985689676283764, 0.02019977190077326,
                    0.03777151472046017, 0.05845449631119356, 0.017956568628560945, -0.07212522602179856,
                    -0.03958975799697206, -0.042927857009029735, -0.025307140867721886]
        self.assertAlmostEqual(np.linalg.norm(uccsd_ansatz.var_params), np.linalg.norm(expected), delta=1e-10)

    def test_uccsd_H2(self):
        """Verify closed-shell UCCSD functionalities for H2."""

        # Build circuit
        uccsd_ansatz = UCCSD(mol_H2_sto3g)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        uccsd_ansatz.update_var_params([5.86665842e-06, 0.0565317429])
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.137270174551959, delta=1e-6)

    def test_uccsd_H4_open(self):
        """Verify open-shell UCCSD functionalities for H4."""

        var_params = [-3.68699814e-03,  1.96987010e-02, -1.12573056e-03,  1.31279980e-04,
                      -4.78466616e-02, -8.56400635e-05,  1.60914422e-03, -2.92334744e-02,
                       3.08405067e-03,  1.39404070e-01,  2.42040971e-02, -2.80714763e-03,
                       1.34675820e-01, -7.70867505e-02,  9.86158364e-05,  2.54962567e-02,
                       5.78169071e-02,  2.46873743e-03, -1.05736505e-01, -4.22089003e-02]

        # Build circuit
        uccsd_ansatz = UCCSD(mol_H4_cation_sto3g)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H4_cation_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        uccsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.639461490, delta=1e-6)

    def test_uccsd_H4_doublecation(self):
        """Verify closed-shell UCCSD functionalities for H4 2+."""

        var_params = [-1.32047062e-02, -7.16419743e-06, -9.25426159e-06,
                       6.84650642e-02,  6.32462456e-02,  1.44675096e-02,
                      -8.34820283e-06, -7.79703747e-06,  3.28660359e-02]

        # Build circuit
        uccsd_ansatz = UCCSD(mol_H4_doublecation_minao)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = load_operator("mol_H4_doublecation_minao_qubitham_jw.data", data_directory=pwd_this_test+"/data", plain_text=True)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        uccsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.854607, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
