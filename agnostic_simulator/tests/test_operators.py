import unittest
import os

from agnostic_simulator import translator, Simulator, Circuit
from agnostic_simulator.helpers import string_ham_to_of, measurement_basis_gates, \
    qubitwise_commutativity_of, exp_value_from_measurement_bases


path_data = os.path.dirname(__file__) + '/data'


class TermsGroupingTest(unittest.TestCase):

    def test_qubitwise_commutativity_of_H2(self):
        """ The JW Pauli hamiltonian of H2 at optimal geometry is a 15-term operator. Using qubitwise-commutativity,
        it is possible to get the expectation value of all 15 terms by only performing measurements in 5 distinct
        measurement bases. This test first verifies the 5 measurement bases have been identified, and then derives
        the expectation value for the qubit Hamiltonian.
        """

        # Load qubit Hamiltonian
        with open(f"{path_data}/H2_qubit_hamiltonian.txt", 'r') as f:
            qb_hamstring = f.read()
        qb_ham = string_ham_to_of(qb_hamstring)

        # Group Hamiltonian terms using qubitwise commutativity
        grouped_ops = qubitwise_commutativity_of(qb_ham, seed=0)

        # Load an optimized quantum circuit (UCCSD) to compute something meaningful in this test
        with open(f"{path_data}/H2_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Only simulate and measure the wavefunction in the required bases (simulator or QPU), store in dict.
        histograms = dict()
        sim = Simulator(target="qulacs")
        for basis, sub_op in grouped_ops.items():
            full_circuit = abs_circ + Circuit(measurement_basis_gates(basis))
            histograms[basis], _ = sim.simulate(full_circuit)

        # Reconstruct exp value of initial input operator using the histograms corresponding to the suboperators
        exp_value = exp_value_from_measurement_bases(grouped_ops, histograms)
        self.assertAlmostEqual(exp_value, sim.get_expectation_value(qb_ham, abs_circ), places=8)

    def test_qubitwise_commutativity_of_H4(self):
        """ Estimating the energy of a rectangle configuration of H4, resulting in a 185-term qubit operator.
        Uses qubitwise-commutativity to identify the number of measurement bases needed (~60) to compute the
        expectation value of the full operator using a pre-loaded UCCSD H4 circuit.
        """

        # Load qubit Hamiltonian
        with open(f"{path_data}/H4_qubit_hamiltonian.txt", 'r') as f:
            qb_hamstring = f.read()
        qb_ham = string_ham_to_of(qb_hamstring)

        # Group Hamiltonian terms using qubitwise commutativity
        grouped_ops = qubitwise_commutativity_of(qb_ham, seed=0)

        # Load an optimized quantum circuit (UCCSD) to compute something meaningful in this test
        with open(f"{path_data}/H4_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Only simulate and measure the wavefunction in the required bases (simulator or QPU), store in dict.
        histograms = dict()
        sim = Simulator(target="qulacs")
        for basis, sub_op in grouped_ops.items():
            full_circuit = abs_circ + Circuit(measurement_basis_gates(basis))
            histograms[basis], _ = sim.simulate(full_circuit)

        # Reconstruct exp value of initial input operator using the histograms corresponding to the suboperators
        exp_value = exp_value_from_measurement_bases(grouped_ops, histograms)
        self.assertAlmostEqual(exp_value, sim.get_expectation_value(qb_ham, abs_circ), places=8)


if __name__ == "__main__":
    unittest.main()
