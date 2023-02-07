# Copyright 2023 Good Chemistry Company.
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

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.qubit_mappings.hcb import hard_core_boson_operator, boson_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.puccd import pUCCD
from tangelo.linq import get_backend


class pUCCDTest(unittest.TestCase):

    def test_puccd_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (keyword,
        list, numpy array).
        """

        puccd_ansatz = pUCCD(mol_H2_sto3g)

        one_ones = np.ones((1,))

        puccd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(puccd_ansatz.var_params, one_ones, decimal=6)

        puccd_ansatz.set_var_params([1.])
        np.testing.assert_array_almost_equal(puccd_ansatz.var_params, one_ones, decimal=6)

        puccd_ansatz.set_var_params(np.array([1.]))
        np.testing.assert_array_almost_equal(puccd_ansatz.var_params, one_ones, decimal=6)

    def test_puccd_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        puccd_ansatz = pUCCD(mol_H2_sto3g)

        self.assertRaises(ValueError, puccd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_puccd_H2(self):
        """Verify closed-shell pUCCD functionalities for H2."""

        # Build circuit.
        puccd_ansatz = pUCCD(mol_H2_sto3g)
        puccd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation.
        qubit_hamiltonian = boson_to_qubit_mapping(
            hard_core_boson_operator(mol_H2_sto3g.fermionic_hamiltonian)
        )

        # Assert energy returned is as expected for given parameters.
        sim = get_backend()
        puccd_ansatz.update_var_params([-0.22617753])
        energy = sim.get_expectation_value(qubit_hamiltonian, puccd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.13727, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
