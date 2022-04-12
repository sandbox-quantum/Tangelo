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

from tangelo.molecule_library import mol_H2_sto3g, mol_H4_cation_sto3g
from tangelo.toolboxes.qubit_mappings import jordan_wigner
from tangelo.toolboxes.ansatz_generator.uccgd import UCCGD
from tangelo.linq import Simulator


class UCCGDTest(unittest.TestCase):

    def test_uccgd_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (keyword,
        list, numpy array).
        """

        uccgd_ansatz = UCCGD(mol_H2_sto3g)

        three_ones = np.ones((3,))

        uccgd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(uccgd_ansatz.var_params, three_ones, decimal=6)

        uccgd_ansatz.set_var_params([1., 1., 1.])
        np.testing.assert_array_almost_equal(uccgd_ansatz.var_params, three_ones, decimal=6)

        uccgd_ansatz.set_var_params(np.array([1., 1., 1.]))
        np.testing.assert_array_almost_equal(uccgd_ansatz.var_params, three_ones, decimal=6)

    def test_uccgd_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        upccgsd_ansatz = UCCGD(mol_H2_sto3g)

        self.assertRaises(ValueError, upccgsd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_uccgd_H2(self):
        """Verify closed-shell UCCGD functionalities for H2."""

        # Build circuit
        uccgd_ansatz = UCCGD(mol_H2_sto3g)
        uccgd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        uccgd_ansatz.update_var_params([0.78525105, 1.14993361, 1.57070471])
        energy = sim.get_expectation_value(qubit_hamiltonian, uccgd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701, delta=1e-6)

    def test_uccgd_H4_open(self):
        """Verify open-shell UCCGD functionalities for H4."""

        var_params = [0.79092606, 0.47506062, 0.80546313, 0.88827205, 1.35659603,
                      0.85509053, 0.86380802, 0.89392798, 0.71922788, 1.36597650,
                      1.41426772, 0.96502797, 0.90349484, 0.58184862, 0.91536039,
                      0.64726991, 0.78419327, 1.59344719, 0.88236760, 1.26174588,
                      0.78342616, 1.21796644, 1.21234109, 0.29929539, 1.53074675,
                      1.41872463, 0.85843055, 1.10503577, 1.19597016, 0.92342705,
                      1.02231599]

        # Build circuit
        uccgd_ansatz = UCCGD(mol_H4_cation_sto3g)
        uccgd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H4_cation_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        uccgd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, uccgd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.64190668, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
