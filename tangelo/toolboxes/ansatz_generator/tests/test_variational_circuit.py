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

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.ansatz_generator.variational_circuit import VariationalCircuitAnsatz


# UCC1 hard coding circuit. Simple example not relying on import Ansatz.
lst_gates = [Gate("RX", 0, parameter=np.pi/2)]
lst_gates += [Gate("H", qubit_i) for qubit_i in range(1, 4)]
lst_gates += [Gate("CNOT", qubit_i+1, qubit_i) for qubit_i in range(3)]
lst_gates += [Gate("RZ", 3, parameter="theta", is_variational=True)]
lst_gates += [Gate("CNOT", qubit_i, qubit_i-1) for qubit_i in range(3, 0, -1)]
lst_gates += [Gate("H", qubit_i) for qubit_i in range(3, 0, -1)]
lst_gates += [Gate("RX", 0, parameter=-np.pi/2)]
circuit = Circuit(lst_gates)


class VariationalCircuitTest(unittest.TestCase):

    def test_init(self):
        """Test initialization of the ansatz class."""
        VariationalCircuitAnsatz(circuit)

    def test_set_var_params(self):
        """Test setting variational parameters."""
        circuit_ansatz = VariationalCircuitAnsatz(circuit)

        single_ones = np.array([1.])

        circuit_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

        circuit_ansatz.set_var_params([1.])
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

        circuit_ansatz.set_var_params(np.array([1.]))
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

    def test_uccsd_incorrect_number_var_params(self):
        """Returns an error if user provide incorrect number of variational parameters."""

        circuit_ansatz = VariationalCircuitAnsatz(circuit)
        self.assertRaises(ValueError, circuit_ansatz.set_var_params, np.array([1., 1.]))


if __name__ == "__main__":
    unittest.main()
