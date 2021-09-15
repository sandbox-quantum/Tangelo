import unittest
import numpy as np

from agnostic_simulator import Gate, Circuit
from qsdk.toolboxes.ansatz_generator.variational_circuit import VariationalCircuitAnsatz


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
        """ Test initialization of the ansatz class. """
        VariationalCircuitAnsatz(circuit)

    def test_set_var_params(self):
        """ Test setting variational parameters. """
        circuit_ansatz = VariationalCircuitAnsatz(circuit)

        single_ones = np.array([1.])

        circuit_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

        circuit_ansatz.set_var_params([1.])
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

        circuit_ansatz.set_var_params(np.array([1.]))
        np.testing.assert_array_almost_equal(circuit_ansatz.var_params,  single_ones, decimal=6)

    def test_uccsd_incorrect_number_var_params(self):
        """ Returns an error if user provide incorrect number of variational parameters. """

        circuit_ansatz = VariationalCircuitAnsatz(circuit)
        self.assertRaises(ValueError, circuit_ansatz.set_var_params, np.array([1., 1.]))


if __name__ == "__main__":
    unittest.main()
