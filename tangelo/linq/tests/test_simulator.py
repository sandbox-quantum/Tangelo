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

"""
    A test class to check that the simulator class functionalities are behaving as expected for the different
    backends and input possible.
"""

import unittest
import os
import time

import numpy as np
from openfermion import load_operator, get_sparse_operator

from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq import Gate, Circuit, get_backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.gate import PARAMETERIZED_GATES
from tangelo.linq.target.backend import Backend, get_expectation_value_from_frequencies_oneterm
from tangelo.helpers.utils import installed_simulator, installed_sv_simulator, installed_backends, installed_clifford_simulators, assert_freq_dict_almost_equal


path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'

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

# Circuit that tests all gates that are supported on all general simulators
init_gates = [Gate('H', 0), Gate('X', 1), Gate('H', 2)]
one_qubit_gate_names = ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "PHASE"]
one_qubit_gates = [Gate(name, target=0) if name not in PARAMETERIZED_GATES else Gate(name, target=0, parameter=0.5)
                   for name in one_qubit_gate_names]
one_qubit_gates += [Gate(name, target=1) if name not in PARAMETERIZED_GATES else Gate(name, target=1, parameter=0.2)
                    for name in one_qubit_gate_names]
two_qubit_gate_names = ["CNOT", "CH", "CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPHASE"]
two_qubit_gates = [Gate(name, target=1, control=0) if name not in PARAMETERIZED_GATES
                   else Gate(name, target=1, control=0, parameter=0.5) for name in two_qubit_gate_names]
swap_gates = [Gate('SWAP', target=[1, 0]), Gate('CSWAP', target=[1, 2], control=0)]
circuit5 = Circuit(init_gates + one_qubit_gates + two_qubit_gates + swap_gates)

# Circuit that tests all gates are supported on clifford simulator
one_qubit_gate_names = ["H", "X", "Y", "Z", "S", "RX", "RY", "RZ"]
one_qubit_gates = [Gate(name, target=0) if name not in PARAMETERIZED_GATES else Gate(name, target=0, parameter=-np.pi/2)
                   for name in one_qubit_gate_names]
one_qubit_gates += [Gate(name, target=1) if name not in PARAMETERIZED_GATES else Gate(name, target=1, parameter=np.pi)
                    for name in one_qubit_gate_names]
clifford_two_qubit_gate_names = ["CNOT", "CX", "CY", "CZ"]
clifford_two_qubit_gates = [Gate(name, target=1, control=0) for name in clifford_two_qubit_gate_names]
circuit_clifford = Circuit(init_gates + one_qubit_gates + clifford_two_qubit_gates + [Gate('SWAP', target=[1, 0])])

# Circuit preparing a mixed-state (i.e. containing a MEASURE instruction in the middle of the circuit)
circuit_mixed = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 1, parameter=-1.), Gate("MEASURE", 0), Gate("X", 0)])
circuit_mixed_1 = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 0, parameter=-1.), Gate("MEASURE", 0), Gate("X", 0)])

# Operators for testing the get_expectation_value functions
op1 = 1.0 * QubitOperator('Z0')  # all Z
op2 = 1.0 * QubitOperator('X1 Y0')  # X and Y
op3 = 1.0 * QubitOperator('Y0') - 2.0 * QubitOperator('Z0 X1')  # Linear combination
op4 = 1.0 * QubitOperator('Z0 Y1 X2')  # Operates on more qubits than circuit size

circuits = [circuit1, circuit2, circuit3, circuit4, circuit5]
ops = [op1, op2, op3]

# Reference results
ref_freqs = list()
ref_freqs.append({'00': 0.5, '10': 0.5})
ref_freqs.append({'01': 0.5, '10': 0.5})
ref_freqs.append({'00': 0.2248275934887, '10': 0.5453235594453, '01': 0.06709898823771, '11': 0.16274985882821})
ref_freqs.append({'00': 1.0})
ref_freqs.append({'000': 0.15972060437359714, '100': 0.2828171838599203, '010': 0.03984122195648572,
                  '110': 0.28281718385992016, '001': 0.15972060437359714, '101': 0.017620989809996816,
                  '011': 0.039841221956485706, '111': 0.01762098980999681})
ref_freqs_clifford = {bs: 0.125 for bs in ['000', '100', '010', '110', '001', '101',  '011', '111']}
reference_exp_values = np.array([[0., 0., 0.], [0., -1., 0.], [-0.41614684, 0.7651474, -1.6096484], [1., 0., 0.],
                                 [-0.20175269, -0.0600213, 1.2972912]])
clifford_reference_exp_values = np.array([0, 0, 2])
reference_mixed = {'01': 0.163, '11': 0.066, '10': 0.225, '00': 0.545}  # With Qiskit noiseless, 1M shots
reference_all = {'101': 0.163, '011': 0.066, '010': 0.225, '100': 0.545}
reference_mid = {'1': 0.7, '0': 0.3}


class TestSimulateAllBackends(unittest.TestCase):

    def test_get_exp_value_operator_too_long(self):
        """ Ensure an error is returned if the qubit operator acts on more qubits than are present in the circuit """
        for b in (installed_simulator | installed_clifford_simulators):
            simulator = get_backend(target=b, n_shots=1)
            self.assertRaises(ValueError, simulator.get_expectation_value, op4, circuit1)

    def test_get_exp_value_empty_operator(self):
        """ If qubit operator is empty, the expectation value is 0 and no computation occurs """
        for b in (installed_simulator | installed_clifford_simulators):
            simulator = get_backend(target=b, n_shots=1)
            exp_value = simulator.get_expectation_value(QubitOperator(), circuit1)
            self.assertTrue(exp_value == 0.)

    def test_get_exp_value_constant_operator(self):
        """ The expectation of the identity term must be 1. """
        for b in (installed_simulator | installed_clifford_simulators):
            simulator = get_backend(target=b, n_shots=1)
            const_op = QubitOperator()
            const_op.terms = {(): 777.}
            exp_value = simulator._get_expectation_value_from_frequencies(const_op, circuit1)
            self.assertTrue(exp_value == 777.)

    def test_simulate_mixed_state(self):
        """ Test mid-circuit measurement (mixed-state simulation) for compatible/testable formats and backends."""

        results = dict()
        for b in installed_simulator:
            sim = get_backend(target=b, n_shots=10**5)
            results[b], _ = sim.simulate(circuit_mixed)
            assert_freq_dict_almost_equal(results[b], reference_mixed, 1e-2)

    def test_simulate_mixed_state_save_measures(self):
        """ Test mid-circuit measurement (mixed-state simulation) for all installed backends."""
        results = dict()
        for b in installed_simulator:
            sim = get_backend(target=b, n_shots=10**3)
            results[b], _ = sim.simulate(circuit_mixed, save_mid_circuit_meas=True)
            assert_freq_dict_almost_equal(results[b], reference_mixed, 8e-2)
            assert_freq_dict_almost_equal(sim.all_frequencies, reference_all, 8e-2)
            assert_freq_dict_almost_equal(sim.mid_circuit_meas_freqs, reference_mid, 8e-2)

    def test_simulate_mixed_state_desired_state(self):
        """ Test mid-circuit measurement (mixed-state simulation) for all installed backends."""

        results = dict()
        exact = {'11': 0.23046888414227926, '10': 0.7695311158577207}
        for b in installed_simulator:
            sim = get_backend(target=b, n_shots=10**3)
            results[b], _ = sim.simulate(circuit_mixed, desired_meas_result="0")
            assert_freq_dict_almost_equal(results[b], exact, 8.e-2)

    def test_desired_meas_len(self):
        """ Test if the desired_meas_result parameter is a string and of the right length."""
        sim = get_backend(target="cirq", n_shots=10**3)
        self.assertRaises(ValueError, sim.simulate, circuit_mixed, desired_meas_result=0)
        self.assertRaises(ValueError, sim.simulate, circuit_mixed, desired_meas_result="01")

    def test_get_exp_value_mixed_state(self):
        """ Test expectation value for mixed-state simulation. Computation done by drawing individual shots."""

        reference = 0.41614683  # Exact value
        results = dict()
        for b in installed_simulator:
            sim = get_backend(target=b, n_shots=10**5)
            results[b] = sim.get_expectation_value(op1, circuit_mixed)
            np.testing.assert_almost_equal(results[b], reference, decimal=2)

    def test_get_variance(self):
        """ Test variance for simple analytical circuit. """

        opx = 1.0 * QubitOperator("X0")
        opy = 1.0 * QubitOperator("Y0")
        opz = 1.0 * QubitOperator("Z0")

        # prepares sqrt(2/3)|0> + -i*sqrt(1/3)|1>
        circuit = Circuit([Gate("RX", 0, parameter=2*np.arcsin(np.sqrt(1/3)))])

        for shots in [None, 10**6]:
            if shots is None:
                precision = 8
            else:
                precision = 2
            sim = get_backend(target='cirq', n_shots=shots)

            # <X> = 0.0, <X^2> = 1.0, so Var(X) = <X^2> - <X>^2 = 1.0
            np.testing.assert_almost_equal(sim.get_variance(opx, circuit), 1.0, decimal=precision)

            # <Y> = -2*sqrt(2)/3, <Y^2> = 1.0, so Var(Y) = <Y^2> - <Y>^2 = 1/9
            np.testing.assert_almost_equal(sim.get_variance(opy, circuit), 1/9, decimal=precision)

            # <Z> = 1/3, <Z^2> = 1.0, so Var(Z) = <Z^2> - <Z>^2 = 8/9
            np.testing.assert_almost_equal(sim.get_variance(opz, circuit), 8/9, decimal=precision)

            # using linearity of variance, Var(<H>) = Var(<X>) + Var(<Y>) + Var(<Z>)
            sum_variance = sim.get_variance(opx + opy + opz, circuit)
            np.testing.assert_almost_equal(sum_variance, 1 + 1/9 + 8/9, decimal=precision)

    def test_get_variance_from_frequencies_oneterm(self):
        """ Test variance given frequencies for one term. """
        op = 1.0 * QubitOperator("Z0")
        for shots in [None, 10**6]:
            if shots is None:
                precision = 8
            else:
                precision = 2
            sim = get_backend(target='cirq', n_shots=shots)
            # <Z> = 1/3, <Z^2> = 1.0, so Var(Z) = <Z^2> - <Z>^2 = 8/9
            frequencies = {'1': 1/3, '0': 2/3}
            oneterm_variance = sim.get_variance_from_frequencies_oneterm(list(op.terms.keys())[0], frequencies)
            np.testing.assert_almost_equal(oneterm_variance, 8/9, decimal=precision)


class TestSimulateStatevector(unittest.TestCase):

    def test_simulate_statevector(self):
        """ Must return correct frequencies for simulation of different quantum circuits with statevector """
        for b in installed_sv_simulator:
            simulator = get_backend(target=b)
            for i, circuit in enumerate(circuits):
                frequencies, _ = simulator.simulate(circuit)
                assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

        for b in installed_clifford_simulators:
            simulator = get_backend(target=b)
            frequencies, _ = simulator.simulate(circuit_clifford)
            assert_freq_dict_almost_equal(ref_freqs_clifford, frequencies, atol=1e-5)

    def test_simulate_mixed_state_desired_statevector(self):
        """ Test mid-circuit measurement (mixed-state simulation) for compatible/testable formats and backends when returning
        a statevector."""

        results = dict()
        results["qulacs"] = np.array([0. + 0.j, 0.87758256 + 0.j, 0. + 0.j, -0.47942554 + 0.j])
        results["qiskit"] = np.array([0. + 0.j, 0.87758256 + 0.j, 0. + 0.j, -0.47942554 + 0.j])
        results["cirq"] = np.array([0. + 0.j, 0. + 0.j, 0.87758256 + 0.j, -0.47942554 + 0.j])
        initial_state = np.array([0, 0, 0, 1])
        freqs_exact = {'10': 0.7701511529340699, '11': 0.2298488470659301}

        for b in installed_sv_simulator:
            sim = get_backend(target=b, n_shots=None)
            f, sv = sim.simulate(circuit_mixed, desired_meas_result="0", return_statevector=True)
            np.testing.assert_array_almost_equal(sv, results[b])
            assert_freq_dict_almost_equal(f, freqs_exact, 1.e-7)
            self.assertAlmostEqual(0.2919265817264289, circuit_mixed.success_probabilities["0"], places=7)

            # Test that initial_statevector is respected
            meas_2_circuit = Circuit([Gate("MEASURE", 0), Gate("MEASURE", 1)])
            f, sv = sim.simulate(meas_2_circuit, desired_meas_result="11",
                                 return_statevector=True, initial_statevector=initial_state)
            np.testing.assert_array_almost_equal(sv, initial_state)
            assert_freq_dict_almost_equal(f, {"11": 1}, 1.e-7)
            self.assertAlmostEqual(1., meas_2_circuit.success_probabilities["11"], places=7)

            # Test that ValueError is raised for desired_meas_result="0" with probability 0. i.e. loop exits successfully
            self.assertRaises(ValueError, sim.simulate, Circuit([Gate("X", 0), Gate("MEASURE", 0)]), True, None, "0")

            sim = get_backend(target=b, n_shots=10**3)
            f, sv = sim.simulate(circuit_mixed, desired_meas_result="0", return_statevector=True)
            np.testing.assert_array_almost_equal(sv, results[b])
            assert_freq_dict_almost_equal(f, freqs_exact, 1.e-1)

    def test_mixed_state_save_measures_return_statevector(self):
        """ Test functionality to return statevector if mid-circuit measurement is saved and n_shots must be 1"""

        sv_exact = {"0": np.array([0.+0.j, 0.76163265-0.64800903j]),
                    "1": np.array([-0.33100336-0.94362958j, 0.+0.j])}
        for b in installed_sv_simulator:
            sim = get_backend(target=b, n_shots=1)
            _, sv = sim.simulate(circuit_mixed_1, save_mid_circuit_meas=True, return_statevector=True)
            np.testing.assert_array_almost_equal(sv, sv_exact[next(iter(sim.mid_circuit_meas_freqs))])

        # Assert raises error when return_statevector=True and n_shots != 1
        for b in installed_sv_simulator:
            sim = get_backend(target=b, n_shots=2)
            self.assertRaises(ValueError, sim.simulate, circuit_mixed_1, True, None, None, True)

    def test_simulate_nshots_from_statevector(self):
        """
            Test the generation of samples following the distribution given by the exact frequencies obtained
            with a statevector simulator. For n_shots high enough, the resulting distribution must approximate
            the exact one.
        """
        for b in installed_sv_simulator:
            simulator = get_backend(target=b, n_shots=10**6)
            for i, circuit in enumerate(circuits):
                frequencies, _ = simulator.simulate(circuit)
                assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-2)

        for b in installed_clifford_simulators:
            simulator = get_backend(target=b, n_shots=10**6)
            frequencies, _ = simulator.simulate(circuit_clifford)
            assert_freq_dict_almost_equal(ref_freqs_clifford, frequencies, atol=1e-2)

    def test_simulate_empty_circuit_from_statevector(self):
        """ Test the generation of frequencies using an initial_statevector and an empty_circuit """
        for b in installed_sv_simulator:
            simulator = get_backend(target=b)
            for i, circuit in enumerate(circuits):
                _, statevector = simulator.simulate(circuit, return_statevector=True)
                frequencies, _ = simulator.simulate(Circuit(n_qubits=circuit.width), initial_statevector=statevector)
                assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-5)

    def test_get_exp_value_from_statevector(self):
        """ Compute the expectation value from the statevector for each statevector backend """
        for b in installed_sv_simulator:
            simulator = get_backend(target=b)
            exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
            for i, circuit in enumerate(circuits):
                for j, op in enumerate(ops):
                    exp_values[i][j] = simulator._get_expectation_value_from_statevector(op, circuit)
            np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

        for b in installed_clifford_simulators:
            simulator = get_backend(target=b)
            clifford_exp_values = np.zeros(len(ops))
            for j, op in enumerate(ops):
                clifford_exp_values[j] = simulator._get_expectation_value_from_statevector(op, circuit_clifford)
            np.testing.assert_almost_equal(clifford_exp_values, clifford_reference_exp_values, decimal=5)

    def test_get_exp_value_from_frequencies_using_initial_statevector(self):
        """ Test the method computing the expectation value from frequencies, with a given simulator
            by generating the statevector first and sampling using an empty state_prep_circuit
        """

        for b in installed_sv_simulator:
            simulator = get_backend(target=b)
            exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
            for i, circuit in enumerate(circuits):
                _, statevector = simulator.simulate(circuit, return_statevector=True)
                for j, op in enumerate(ops):
                    exp_values[i][j] = simulator._get_expectation_value_from_frequencies(op,
                                                                                         Circuit(n_qubits=circuit.width),
                                                                                         initial_statevector=statevector)
            np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

    def test_get_exp_value_from_statevector_h2(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H2 sto-3g = [("H", (0., 0., 0.)), ("H", (0., 0., 0.741377))]
        """
        qubit_operator = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

        with open(f"{path_data}/H2_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()

        abs_circ = translate_c(openqasm_circ, "tangelo", source="openqasm")
        expected = -1.1372704
        test_fail = False

        for b in installed_sv_simulator:
            sim = get_backend(target=b)
            tstart = time.time()
            energy = sim.get_expectation_value(qubit_operator, abs_circ)
            tstop = time.time()
            print(f"H2 get exp value with {b:10s} returned {energy:.7f} \t Elapsed: {tstop - tstart:.3f} s.")

            try:
                self.assertAlmostEqual(energy, expected, delta=1e-5)
            except AssertionError:
                test_fail = True
                print(f"{self._testMethodName} : Assertion failed {b} (result = {energy:.7f}, expected = {expected})")
        if test_fail:
            assert False

    def test_get_exp_value_from_initial_statevector_h2(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H2 sto-3g = [("H", (0., 0., 0.)), ("H", (0., 0., 0.741377))]
            Generate statevector first and then get_expectation value from statevector and empty circuit.
        """
        qubit_operator = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

        with open(f"{path_data}/H2_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()

        abs_circ = translate_c(openqasm_circ, "tangelo", source="openqasm")
        expected = -1.1372704
        test_fail = False

        for b in installed_sv_simulator:
            sim = get_backend(target=b)
            tstart = time.time()
            _, statevector = sim.simulate(abs_circ, return_statevector=True)
            energy = sim.get_expectation_value(qubit_operator, Circuit(n_qubits=abs_circ.width),
                                               initial_statevector=statevector)
            tstop = time.time()
            print(f"H2 get exp value with {b:10s} returned {energy:.7f} \t Elapsed: {tstop - tstart:.3f} s.")

            try:
                self.assertAlmostEqual(energy, expected, delta=1e-5)
            except AssertionError:
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
        qubit_operator = load_operator("mol_H4_qubitham.data", data_directory=path_data, plain_text=True)

        with open(f"{path_data}/H4_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()

        abs_circ = translate_c(openqasm_circ, "tangelo", source="openqasm")
        expected = -1.9778374
        test_fail = False

        for b in installed_sv_simulator:
            sim = get_backend(target=b)
            tstart = time.time()
            energy = sim.get_expectation_value(qubit_operator, abs_circ)
            tstop = time.time()
            print(f"H4 get exp value with {b:10s} returned {energy:.7f} \t Elapsed: {tstop - tstart:.3f} s.")

            try:
                self.assertAlmostEqual(energy, expected, delta=1e-5)
            except AssertionError:
                test_fail = True
                print(f"{self._testMethodName} : Assertion failed {b} (result = {energy:.7f}, expected = {expected})")
        if test_fail:
            assert False

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_get_exp_value_from_statevector_with_shots_h2(self):
        """ Get expectation value of large circuits and qubit Hamiltonians corresponding to molecules.
            Molecule: H2 sto-3g = [("H", (0., 0., 0.)), ("H", (0., 0., 0.741377))]
            The result is computed using samples ("shots") drawn form a statevector simulator here. This is the kind
            of results we could expect from a noiseless QPU.
        """
        qubit_operator = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

        with open(f"{path_data}/H2_UCCSD.qasm", "r") as circ_handle:
            openqasm_circ = circ_handle.read()
        abs_circ = translate_c(openqasm_circ, "tangelo", source="openqasm")

        simulator = get_backend(target="qulacs", n_shots=10**6)
        expected = -1.1372704

        energy = simulator.get_expectation_value(qubit_operator, abs_circ)
        self.assertAlmostEqual(energy, expected, delta=1e-3)

    def test_get_exp_value_mixed_state_desired_measurement_with_shots(self):
        """ Get expectation value of mixed state by post-selecting on desired measurement."""
        qubit_operator = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1") + QubitOperator("X0 Y1", 1j)

        ham = get_sparse_operator(qubit_operator.to_openfermion()).toarray()
        exact_sv = np.array([0.+0.j, 0.+0.j, 0.87758256+0.j, -0.47942554+0.j])
        exact_exp = np.vdot(exact_sv, ham @ exact_sv)

        simulator = get_backend(n_shots=10**4)
        sim_exp = simulator.get_expectation_value(qubit_operator, circuit_mixed, desired_meas_result="0")
        self.assertAlmostEqual(exact_exp, sim_exp, delta=1.e-1)

    def test_get_exp_value_empty_circuit(self):
        """ If the circuit is empty and we have a non-zero number of qubits, frequencies just only show all-|0> state
        observed and compute the expectation value using these frequencies """

        empty_circuit = Circuit([], n_qubits=2)
        identity_circuit = Circuit([Gate('X', 0), Gate('X', 1)] * 2)

        for b in (installed_sv_simulator | installed_clifford_simulators):
            simulator = get_backend(target=b)
            for op in [op1, op2]:
                exp_value_empty = simulator.get_expectation_value(op, empty_circuit)
                exp_value_identity = simulator.get_expectation_value(op, identity_circuit)
                np.testing.assert_almost_equal(exp_value_empty, exp_value_identity, decimal=8)

    def test_get_exp_value_complex(self):
        """ Get expectation value of qubit operator with complex coefficients """

        for b in installed_sv_simulator:
            simulator = get_backend(target=b)

            # Return complex expectation value corresponding to linear combinations of real and imaginary parts
            op_c = op1 + 1.0j * op2
            exp_c = simulator.get_expectation_value(op_c, circuit3)
            exp_r1 = simulator.get_expectation_value(op1, circuit3)
            exp_r2 = simulator.get_expectation_value(op2, circuit3)
            self.assertAlmostEqual(exp_c.real, exp_r1, delta=1.e-12)
            self.assertAlmostEqual(exp_c.imag, exp_r2, delta=1.e-12)

            # Edge case: all coefficients are complex but with imaginary part null: exp value must return a float
            op_c = op1 + 0.j * op1
            exp_c = simulator.get_expectation_value(op_c, circuit3)
            assert (type(exp_c) in {float, np.float64} and exp_c == exp_r1)

    def test_get_exp_value_from_frequencies(self):
        """ Test the method computing the expectation value from frequencies, with a given simulator """

        for b in installed_sv_simulator:
            simulator = get_backend(target=b)
            exp_values = np.zeros((len(circuits), len(ops)), dtype=float)
            for i, circuit in enumerate(circuits):
                for j, op in enumerate(ops):
                    exp_values[i][j] = simulator._get_expectation_value_from_frequencies(op, circuit)
            np.testing.assert_almost_equal(exp_values, reference_exp_values, decimal=5)

        for b in installed_clifford_simulators:
            simulator = get_backend(target=b)
            clifford_exp_values = np.zeros(len(ops))
            for j, op in enumerate(ops):
                clifford_exp_values[j] = simulator._get_expectation_value_from_frequencies(op, circuit_clifford)
            np.testing.assert_almost_equal(clifford_exp_values, clifford_reference_exp_values, decimal=5)


class TestSimulateMisc(unittest.TestCase):
    @unittest.skipIf("qdk" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_n_shots_needed(self):
        """
            Raise an error if user chooses a target backend that does not provide access to a statevector and
            also does not provide a number of shots for the simulation.
        """
        self.assertRaises(ValueError, get_backend, target="qdk")

    @unittest.skipIf("qdk" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_simulate_qdk(self):
        """
            Must return correct frequencies for simulation of different quantum circuits.
            The accuracy is correlated to the number of shots taken in the simulation.
            Backend: qdk.
        """
        simulator = get_backend(target="qdk", n_shots=10**4)
        for i, circuit in enumerate(circuits):
            frequencies, _ = simulator.simulate(circuit)
            assert_freq_dict_almost_equal(ref_freqs[i], frequencies, atol=1e-1)

    @unittest.skipIf("qdk" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_get_exp_value_from_frequencies_qdk(self):
        """ Test specific to QDK to ensure results are not impacted by code specific to frequency computation
            as well as the recompilation of the Q# file used in successive simulations """

        simulator = get_backend(target="qdk", n_shots=10**4)
        exp_values = np.zeros((len(ops)), dtype=float)
        for j, op in enumerate(ops):
            exp_values[j] = simulator.get_expectation_value(op, circuit3)
        np.testing.assert_almost_equal(exp_values, reference_exp_values[2], decimal=1)

    def test_get_exp_value_from_frequencies_oneterm(self):
        """ Test static method computing the expectation value of one term, when the results of a simulation
         are being provided as input. """

        term, coef = ((0, 'Z'),), 1.0  # Data as presented in Openfermion's QubitOperator.terms attribute
        exp_value = coef * get_expectation_value_from_frequencies_oneterm(term, ref_freqs[2])
        np.testing.assert_almost_equal(exp_value, -0.41614684, decimal=5)

    def test_invalid_target(self):
        """ Ensure an error is returned if the target simulator is not supported."""
        self.assertRaises(ValueError, get_backend, 'banana')

    def test_user_provided_simulator(self):
        """Test user defined target simulator that disregards the circuit gates and only returns zero state or one state"""

        class TrueFalseSimulator(Backend):
            def __init__(self, n_shots=None, noise_model=None, return_zeros=True):
                """Instantiate simulator object that always returns all zeros or all ones ignoring circuit operations."""
                super().__init__(n_shots=n_shots, noise_model=noise_model)
                self.return_zeros = return_zeros

            def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
                """Perform state preparation corresponding self.return_zeros."""

                statevector = np.zeros(2**source_circuit.width, dtype=complex)
                if self.return_zeros:
                    statevector[0] = 1.
                else:
                    statevector[-1] = 1.

                frequencies = self._statevector_to_frequencies(statevector)

                return (frequencies, np.array(statevector)) if return_statevector else (frequencies, None)

            @staticmethod
            def backend_info():
                return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": False}

        sim = get_backend(TrueFalseSimulator, n_shots=1, noise_model=None, return_zeros=True)
        f, sv = sim.simulate(circuit1, return_statevector=True)
        assert_freq_dict_almost_equal(f, {"00": 1}, 1.e-7)
        np.testing.assert_almost_equal(np.array([1., 0., 0., 0.]), sv)
        self.assertAlmostEqual(sim.get_expectation_value(QubitOperator("Z0", 1.), circuit1), 1.)

        sim = get_backend(TrueFalseSimulator, n_shots=1, noise_model=None, return_zeros=False)
        f, sv = sim.simulate(circuit1, return_statevector=True)
        assert_freq_dict_almost_equal(f, {"11": 1}, 1.e-7)
        np.testing.assert_almost_equal(np.array([0., 0., 0., 1.]), sv)
        self.assertAlmostEqual(sim.get_expectation_value(QubitOperator("Z0", 1.), circuit1), -1.)


if __name__ == "__main__":
    unittest.main()
