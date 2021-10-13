import unittest
import os

from qsdk.helpers.utils import default_simulator
from qsdk.backendbuddy import translator, Simulator, Circuit
from qsdk.backendbuddy.helpers import string_ham_to_of, measurement_basis_gates
from qsdk.toolboxes.operators import QubitOperator
from qsdk.toolboxes.measurements import get_measurement_estimate

path_data = os.path.dirname(__file__) + '/data'

op1 = 1. * QubitOperator('X1 Y0')
op2 = 0.01 * QubitOperator('Z0 X1')


class MeasurementsTest(unittest.TestCase):

    def test_measurement_uniform(self):
        """ Test uniform measurement estimation method on simple cases """

        # Integer log10 coefficients
        mes1 = get_measurement_estimate(op1)
        mes2 = get_measurement_estimate(op2)
        mes3 = get_measurement_estimate(op2 + 0.1*op1)

        # Float log10 coefficients
        mes4 = get_measurement_estimate(0.6*op1)

        # Coefficient smaller than desired accuracy -> 0
        mes5 = get_measurement_estimate(0.0001*op1)

        # Borderline case: Instead of 1 measurement, take 100.
        mes6 = get_measurement_estimate(0.001*op1)

        # Adjust digits
        mes7 = get_measurement_estimate(op1, digits=2)

        assert(mes1 == {((0, 'Y'), (1, 'X')): 100000000})
        assert(mes2 == {((0, 'Z'), (1, 'X')): 10000})
        assert(mes3 == {((0, 'Z'), (1, 'X')): 10000, ((0, 'Y'), (1, 'X')): 1000000})
        assert(mes4 == {((0, 'Y'), (1, 'X')): 36000000})
        assert(mes5 == {((0, 'Y'), (1, 'X')): 0})
        assert(mes6 == {((0, 'Y'), (1, 'X')): 100})
        assert(mes7 == {((0, 'Y'), (1, 'X')): 1000000})

    def test_measurement_uniform_H2(self):
        """ Test on UCCSD H2 usecase that uniform measurement estimation method guarantees on average the level
        of accuracy expected by the user """

        # Load state preparation circuit
        with open(f"{path_data}/H2_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Load qubit Hamiltonian
        with open(f"{path_data}/H2_qubit_hamiltonian.txt", 'r') as f:
            qb_hamstring = f.read()
        qb_ham = string_ham_to_of(qb_hamstring)

        # Get exact expectation value using a simulator
        sim_exact = Simulator()
        freqs_exact, _ = sim_exact.simulate(abs_circ)
        exp_val_exact = sim_exact.get_expectation_value(qb_ham, abs_circ)

        # Get measurement estimate using "uniform" method
        mes_dict = get_measurement_estimate(qb_ham, digits=2)

        # Get samples and compute expectation value from measurements, using exact state preparation. Compute error.
        diffs = []
        n_repeat = 50
        for i in range(n_repeat):
            exp_val = 0.
            sim = Simulator(target=default_simulator, n_shots=1)
            for m_basis, coef in qb_ham.terms.items():
                term_circ = abs_circ + Circuit(measurement_basis_gates(m_basis))
                sim.n_shots = mes_dict[m_basis]
                freqs, _ = sim.simulate(term_circ)
                exp_val += sim.get_expectation_value_from_frequencies_oneterm(m_basis, freqs) * coef
            diffs.append(abs(exp_val - exp_val_exact))

        # Check that on average, we deliver the expected accuracy
        average_diff = sum(diffs)/n_repeat
        assert(average_diff <= 1e-2)


if __name__ == "__main__":
    unittest.main()
