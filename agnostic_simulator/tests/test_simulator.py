"""
    A test class to check that the simulator class functionalities are behaving as expected for the different
    backends and input possible.
"""

import unittest
import os
import time
import numpy as np
from openfermion.ops import QubitOperator

from agnostic_simulator import Gate, Circuit, translator, Simulator, backend_info
from agnostic_simulator.helpers import string_ham_to_of

path_data = os.path.dirname(__file__) + '/data'

# Simple circuit for superposition, also tells us qubit ordering as well immediately from the statevector
# probabilities : |00> = 0.5  |01> = 0.5
circuit1 = Circuit([Gate("H", 0)], n_qubits=2)
# 2-qubit circuit checking all the basic gates that are not defined up to a convention (e.g unambiguous)
mygates = [Gate("H", 0), Gate("S", 0), Gate("X", 0), Gate("T", 1), Gate("Y", 1), Gate("Z", 1)]
mygates += [Gate("CNOT", 1, control=0)]
circuit2 = Circuit(mygates)
# Circuit for the parametrized rotation gates Rx and Ry. Some convention about the sign of theta or a phase may appear
circuit3 = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 1, parameter=-1.)])
# Circuit for the parametrized rotation gate Rz. Some convention about the sign of theta or a phase may appear
circuit4 = Circuit([Gate("RZ", 0, parameter=2.)], n_qubits=2)
# Circuit preparing a mixed-state (e.g containing a MEASURE instruction in the middle of the circuit)
circuit_mixed = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 1, parameter=-1.), Gate("MEASURE", 0), Gate("X", 0)])

# Operators for testing the get_expectation_value functions
op1 = 1.0 * QubitOperator('Z0')  # all Z
op2 = 1.0 * QubitOperator('X1 Y0')  # X and Y
op3 = 1.0 * QubitOperator('Y0') - 2.0 * QubitOperator('Z0 X1')  # Linear combination
op4 = 1.0 * QubitOperator('Z0 Y1 X2')  # Operates on more qubits than circuit size

circuits = [circuit1, circuit2, circuit3, circuit4]
ops = [op1, op2, op3]

# Reference results
ref_freqs = list()
ref_freqs.append({'00': 0.5, '10': 0.5})
ref_freqs.append({'01': 0.5, '10': 0.5})
ref_freqs.append({'00': 0.2248275934887, '10': 0.5453235594453, '01': 0.06709898823771, '11': 0.16274985882821})
ref_freqs.append({'00': 1.0})
reference_exp_values = np.array([[0., 0., 0.], [0., -1., 0.], [-0.41614684, 0.7651474, -1.6096484], [1., 0., 0.]])
reference_mixed = {'01': 0.163, '11': 0.066, '10': 0.225, '00': 0.545}  # With Qiskit noiseless, 1M shots


def assert_freq_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two frequency dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Dictionary entries beyond tolerance {atol}: \n{d1} \n{d2}")
    return True


class TestSimulate(unittest.TestCase):

    def test_n_shots_needed(self):
        """
            Raise an error if user chooses a target backend that does not provide access to a statevector and
            also does not provide a number of shots for the simulation.
        """
        self.assertRaises(ValueError, Simulator, target="qdk")

    def test_simulate_qulacs(self):
        """
            Must return correct frequencies for simulation of different quantum circuits
            Backend: qulacs
        """
        simulator = Simulator(target="qulacs")
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

    def test_simulate_qiskit(self):
        """
            Must return correct frequencies for simulation of different quantum circuits
            Backend: qiskit
        """
        simulator = Simulator(target="qiskit")
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

    def test_simulate_projectq(self):
        """
            Must return correct frequencies for simulation of different quantum circuits
            Backend: Projectq
        """
        simulator = Simulator(target="projectq")
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

    def test_simulate_cirq(self):
        """
            Must return correct frequencies for simulation of different quantum circuits
            Backend: cirq
        """
        simulator = Simulator(target="cirq")
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

    def test_simulate_qdk(self):
        """
            Must return correct frequencies for simulation of different quantum circuits.
            The accuracy is correlated to the number of shots taken in the simulation.
            Backend: qdk.
        """
        simulator = Simulator(target="qdk", n_shots=10**4)
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-1)

    def test_simulate_nshots_from_statevector(self):
        """
            Test the generation of samples following the distribution given by the exact frequencies obtained
            with a statevector simulator. For n_shots high enough, the resulting distribution must approximate
            the exact one.
        """
        simulator = Simulator(target="qulacs", n_shots=10 ** 6)
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-2)

    def test_get_exp_value_operator_too_long(self):
        """ Ensure an error is returned if the qubit operator acts on more qubits than are present in the circuit """
        for b in backend_info:
            simulator = Simulator(target=b, n_shots=1)
            self.assertRaises(ValueError, simulator.get_expectation_value, op4, circuit1)

    def test_get_exp_value_empty_operator(self):
        """ If qubit operator is empty, the expectation value is 0 and no computation occurs """
        for b in backend_info:
            simulator = Simulator(target=b, n_shots=1)
            exp_value = simulator.get_expectation_value(QubitOperator(), circuit1)
            self.assertTrue(exp_value == 0.)

    def test_get_exp_value_constant_operator(self):
        """ The expectation of the identity term must be 1. """
        for b in backend_info:
            simulator = Simulator(target=b, n_shots=1)
            const_op = QubitOperator()
            const_op.terms = {(): 777.}
            exp_value = simulator._get_expectation_value_from_frequencies(const_op, circuit1)
            self.assertTrue(exp_value == 777.)

    def test_get_exp_value_empty_circuit(self):
        """ If the circuit is empty and we have a non-zero number of qubits, frequencies just only show all-|0> state
        observed and compute the expectation value using these frequencies """

        empty_circuit = Circuit([], n_qubits=2)
        identity_circuit = Circuit([Gate('X', 0), Gate('X', 1)] * 2)

        for b in ['qulacs', 'qiskit', 'projectq', 'cirq']:
            simulator = Simulator(target=b)
            for op in [op1, op2]:
                exp_value_empty = simulator.get_expectation_value(op, empty_circuit)
                exp_value_identity = simulator.get_expectation_value(op, identity_circuit)
                np.testing.assert_almost_equal(exp_value_empty, exp_value_identity, decimal=8)

    def test_get_exp_value_from_statevector_qiskit(self):
        """ Test the generic method computing the expectation value from a statevector with a simulator providing
            a statevector """

        simulator = Simulator(target="qiskit")
        exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
        for i, circuit in enumerate(circuits):
            for j, op in enumerate(ops):
                exp_values[i][j] = simulator._get_expectation_value_from_statevector(op, circuit)
        np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_from_statevector_cirq(self):
        """ Test the generic method computing the expectation value from a statevector with a simulator providing
            a statevector """

        simulator = Simulator(target="cirq")
        exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
        for i, circuit in enumerate(circuits):
            for j, op in enumerate(ops):
                exp_values[i][j] = float(simulator._get_expectation_value_from_statevector(op, circuit))
        np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_from_statevector_qulacs(self):
        """ Use the fast qulacs built-in method computing the expectation value from a qulacs state """

        simulator = Simulator(target="qulacs")
        exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
        for i, circuit in enumerate(circuits):
            for j, op in enumerate(ops):
                exp_values[i][j] = simulator._get_expectation_value_from_statevector(op, circuit)
        np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_from_statevector_projectq(self):
        """ Use the fast projectq built-in method computing the expectation value from a statevector """

        simulator = Simulator(target="projectq")
        exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
        for i, circuit in enumerate(circuits):
            for j, op in enumerate(ops):
                exp_values[i][j] = simulator._get_expectation_value_from_statevector(op, circuit)
        np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_complex(self):
        """ Get expectation value of qubit operator with complex coefficients """

        for b in ["qulacs", "qiskit", "projectq", "cirq"]:
            simulator = Simulator(target=b)

            # Return complex expectation value corresponding to linear combinations of real and imaginary parts
            op_c = op1 + 1.0j * op2
            exp_c = simulator.get_expectation_value(op_c, circuit3)
            exp_r1 = simulator.get_expectation_value(op1, circuit3)
            exp_r2 = simulator.get_expectation_value(op2, circuit3)
            print(f"complex exp with {b} ::\t {exp_c} =? {exp_r1} + {exp_r2}j")
            assert(exp_c == (exp_r1 + 1.0j * exp_r2))

            # Edge case: all coefficients are complex but with imaginary part null: exp value must return a float
            op_c = op1 + 0.j * op1
            exp_c = simulator.get_expectation_value(op_c, circuit3)
            assert (type(exp_c) in {float, np.float64} and exp_c == exp_r1)

    def test_get_exp_value_from_statevector_h2(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H2 sto-3g = [("H", (0., 0., 0.)), ("H", (0., 0., 0.741377))]
        """
        with open(f"{path_data}/H2_qubit_hamiltonian.txt", "r") as ham_handle:
            string_ham = ham_handle.read()
            qubit_operator = string_ham_to_of(string_ham)

        with open(f"{path_data}/H2_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()

        abs_circ = translator._translate_openqasm2abs(openqasm_circ)
        backends = ["qulacs", "projectq", "qiskit", "cirq"]
        results = dict()
        expected = -1.1372704
        test_fail = False

        for b in backends:
            sim = Simulator(target=b)
            tstart = time.time()
            energy = sim.get_expectation_value(qubit_operator, abs_circ)
            tstop = time.time()
            print(f"H2 get exp value with {b:10s} returned {energy:.7f} \t Elapsed: {tstop-tstart:.3f} s.")

            try:
                self.assertAlmostEqual(energy, expected, delta=1e-5)
            except:
                test_fail = True
                print(f"{self._testMethodName} : Assertion failed {b} (result = {energy:.7f}, expected = {expected})")
        if test_fail:
            assert False

    def test_get_exp_value_from_statevector_h4(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H4 sto-3g
            H4 = [['H', [0.7071067811865476,   0.0,                 0.0]],
                  ['H', [0.0,                  0.7071067811865476,  0.0]],
                  ['H', [-1.0071067811865476,  0.0,                 0.0]],
                  ['H', [0.0,                 -1.0071067811865476,  0.0]]]
        """

        with open(f"{path_data}/H4_qubit_hamiltonian.txt", "r") as ham_handle:
            string_ham = ham_handle.read()
            qubit_operator = string_ham_to_of(string_ham)
        with open(f"{path_data}/H4_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()

        abs_circ = translator._translate_openqasm2abs(openqasm_circ)
        backends = ["qulacs", "projectq", "qiskit", "cirq"]
        results = dict()
        expected = -1.9778374
        test_fail = False

        for b in backends:
            sim = Simulator(target=b)
            tstart = time.time()
            energy = sim.get_expectation_value(qubit_operator, abs_circ)
            tstop = time.time()
            print(f"H4 get exp value with {b:10s} returned {energy:.7f} \t Elapsed: {tstop - tstart:.3f} s.")

            try:
                self.assertAlmostEqual(energy, expected, delta=1e-5)
            except:
                test_fail = True
                print(f"{self._testMethodName} : Assertion failed {b} (result = {energy:.7f}, expected = {expected})")
        if test_fail:
            assert False

    def test_get_exp_value_from_statevector_with_shots_h2(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H2 sto-3g = [("H", (0., 0., 0.)), ("H", (0., 0., 0.741377))]
            The result is computed using samples ("shots") drawn form a statevector simulator here. This is the kind
            of results we could expect from a noiseless QPU.
        """
        with open(f"{path_data}/H2_qubit_hamiltonian.txt", "r") as ham_handle:
            string_ham = ham_handle.read()
            qubit_operator = string_ham_to_of(string_ham)
        with open(f"{path_data}/H2_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        simulator = Simulator(target="qulacs", n_shots=10**6)
        expected = -1.1372704

        energy = simulator.get_expectation_value(qubit_operator, abs_circ)
        self.assertAlmostEqual(energy, expected, delta=1e-3)

    def test_get_exp_value_from_frequencies(self):
        """ Test the method computing the expectation value from frequencies, with a given simulator """

        simulator = Simulator(target="qulacs")
        exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
        for i, circuit in enumerate(circuits):
            for j, op in enumerate(ops):
                exp_values[i][j] = simulator._get_expectation_value_from_frequencies(op, circuit)
        np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_from_frequencies_qdk(self):
        """ Test specific to QDK to ensure results are not impacted by code specific to frequency computation
            as well as the recompilation of the Q# file used in successive simulations """

        simulator = Simulator(target="qdk", n_shots=10**4)
        exp_values = np.zeros((len(ops)), dtype=float)
        for j, op in enumerate(ops):
            exp_values[j] = simulator.get_expectation_value(op, circuit3)
        np.testing.assert_almost_equal(exp_values, reference_exp_values[2], decimal=1)

    def test_get_exp_value_from_frequencies_oneterm(self):
        """ Test static method computing the expectation value of one term, when the results of a simulation
         are being provided as input. """

        term, coef = ((0, 'Z'),), 1.0  # Data as presented in Openfermion's QubitOperator.terms attribute
        exp_value = coef * Simulator.get_expectation_value_from_frequencies_oneterm(term, ref_freqs[2])
        np.testing.assert_almost_equal(exp_value, -0.41614684, decimal=5)

    def test_simulate_mixed_state(self):
        """ Test mid-circuit measurement (mixed-state simulation) for compatible/testable formats and backends.
        Mixed-state do not have a statevector representation, as they are a statistical mixture of several statevectors.
        Simulating individual shots is suitable,

        Some simulators are NOT good at this, by design (ProjectQ).
        """

        backends = ["qiskit", "qulacs", "projectq", "qdk", "cirq"]
        results = dict()
        for b in backends:
            sim = Simulator(target=b, n_shots=10**5)
            results[b], _ = sim.simulate(circuit_mixed)
            assert_freq_dict_almost_equal(results[b], reference_mixed, 1e-2)

    def test_get_exp_value_mixed_state(self):
        """ Test expectation value for mixed-state simulation. Computation done by drawing individual shots.
        Some simulators are NOT good at this, by design (ProjectQ). """

        reference = 0.41614683  # Exact value
        backends = ["qiskit", "qulacs", "projectq", "qdk", "cirq"]
        results = dict()
        for b in backends:
            sim = Simulator(target=b, n_shots=10**5)
            results[b] = sim.get_expectation_value(op1, circuit_mixed)
            np.testing.assert_almost_equal(results[b], reference, decimal=2)


if __name__ == "__main__":
    unittest.main()
