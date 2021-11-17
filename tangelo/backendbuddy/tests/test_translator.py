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

"""
    A test class to check that the translation process yields a circuit identical to the native one for the
    different backends supported
"""

import unittest
import os
import numpy as np

from tangelo.backendbuddy import Gate, Circuit
import tangelo.backendbuddy.translator as translator
from tangelo.helpers.utils import installed_backends

path_data = os.path.dirname(os.path.realpath(__file__)) + '/data'

gates = [Gate("H", 2), Gate("CNOT", 1, control=0), Gate("CNOT", 2, control=1), Gate("Y", 0), Gate("S", 0)]
abs_circ = Circuit(gates) + Circuit([Gate("RX", 1, parameter=2.)])
references = [0., 0.38205142 ** 2, 0., 0.59500984 ** 2, 0., 0.38205142 ** 2, 0., 0.59500984 ** 2]

abs_circ_mixed = Circuit(gates) + Circuit([Gate("RX", 1, parameter=1.5), Gate("MEASURE", 0)])


class TestTranslation(unittest.TestCase):

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_qulacs(self):
        """
            Compares the results of a simulation with Qulacs using a qulacs circuit directly
            VS one obtained through translation from an abstract format
        """
        import qulacs

        # Generates the qulacs circuit by translating from the abstract one
        translated_circuit = translator.translate_qulacs(abs_circ)

        # Run the simulation
        state1 = qulacs.QuantumState(abs_circ.width)
        translated_circuit.update_quantum_state(state1)

        # Directly define the same circuit through qulacs
        # NB: this includes convention fixes for some parametrized rotation gates (-theta instead of theta)
        qulacs_circuit = qulacs.QuantumCircuit(3)
        qulacs_circuit.add_H_gate(2)
        qulacs_circuit.add_CNOT_gate(0, 1)
        qulacs_circuit.add_CNOT_gate(1, 2)
        qulacs_circuit.add_Y_gate(0)
        qulacs_circuit.add_S_gate(0)
        qulacs_circuit.add_RX_gate(1, -2.)  # Convention: sign of theta

        # Run the simulation
        state2 = qulacs.QuantumState(abs_circ.width)
        qulacs_circuit.update_quantum_state(state2)

        # Assert that both simulations returned the same state vector
        np.testing.assert_array_equal(state1.get_vector(), state2.get_vector())

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_qiskit(self):
        """
            Compares the results of a simulation with Qulacs using a qulacs circuit directly
            VS one obtained through translation from an abstract format
        """

        import qiskit

        # Generate the qiskit circuit by translating from the abstract one and print it
        translated_circuit = translator.translate_qiskit(abs_circ)

        # Generate the qiskit circuit directly and print it
        circ = qiskit.QuantumCircuit(3, 3)
        circ.h(2)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.y(0)
        circ.s(0)
        circ.rx(2., 1)

        # Simulate both circuits, assert state vectors are equal
        qiskit_simulator = qiskit.Aer.get_backend("aer_simulator", method='statevector') 
        save_state_circuit = qiskit.QuantumCircuit(abs_circ.width, abs_circ.width)
        save_state_circuit.save_statevector()
        translated_circuit = translated_circuit.compose(save_state_circuit)
        circ = circ.compose(save_state_circuit)

        sim_results = qiskit_simulator.run(translated_circuit).result()
        v1 = sim_results.get_statevector(translated_circuit)

        sim_results = qiskit_simulator.run(circ).result()
        v2 = sim_results.get_statevector(circ)

        np.testing.assert_array_equal(v1, v2)

    @unittest.skipIf("cirq" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_cirq(self):
        """
            Compares the results of a simulation with cirq using a cirq circuit directly
            VS one obtained through translation from an abstract format
        """

        import cirq

        # Generate the qiskit circuit by translating from the abstract one and print it
        translated_circuit = translator.translate_cirq(abs_circ)

        # Generate the cirq circuit directly and print it
        qubit_labels = cirq.LineQubit.range(3)
        circ = cirq.Circuit()
        circ.append(cirq.H(qubit_labels[2]))
        circ.append(cirq.CNOT(qubit_labels[0], qubit_labels[1]))
        circ.append(cirq.CNOT(qubit_labels[1], qubit_labels[2]))
        circ.append(cirq.Y(qubit_labels[0]))
        circ.append(cirq.S(qubit_labels[0]))
        gate_rx = cirq.rx(2.)
        circ.append(gate_rx(qubit_labels[1]))

        # Simulate both circuits, assert state vectors are equal
        cirq_simulator = cirq.Simulator()

        job_sim = cirq_simulator.simulate(circ)
        v1 = job_sim.final_state_vector

        job_sim = cirq_simulator.simulate(translated_circuit)
        v2 = job_sim.final_state_vector

        np.testing.assert_array_equal(v1, v2)

    @unittest.skipIf("qdk" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_qdk(self):
        """ Compares the frequencies computed by the QDK/Q# shot-based simulator to the theoretical ones """

        # Generate the qdk circuit by translating from the abstract one and print it
        translated_circuit = translator.translate_qsharp(abs_circ)
        print(translated_circuit)

        # Write to file
        with open('tmp_circuit.qs', 'w+') as f_out:
            f_out.write(translated_circuit)

        # Compile all qsharp files found in directory and import the qsharp operation
        import qsharp
        qsharp.reload()
        from MyNamespace import EstimateFrequencies

        # Simulate, return frequencies
        n_shots = 10**5
        probabilities = EstimateFrequencies.simulate(nQubits=abs_circ.width, nShots=n_shots)
        print("Q# frequency estimation with {0} samples: \n {1}".format(n_shots, probabilities))

        # Compares with theoretical probabilities obtained through a statevector simulator
        np.testing.assert_almost_equal(np.array(probabilities), np.array(references), 2)

    @unittest.skipIf("projectq" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_projectq(self):
        """ Compares state vector of native ProjectQ circuit against translated one """
        from projectq.ops import All, Measure, H, CX, Y, S, Rx
        from projectq import MainEngine

        translated_circuit = translator.translate_projectq(abs_circ)
        instructions = translated_circuit.split("\n")

        eng = MainEngine()
        Qureg = eng.allocate_qureg(abs_circ.width)
        for instruction in instructions[abs_circ.width:]:
            exec(instruction)
        eng.flush()
        _, v1 = eng.backend.cheat()
        All(Measure) | Qureg
        eng.flush()

        # # Native ProjectQ circuit
        eng2 = MainEngine()
        Qureg2 = eng2.allocate_qureg(3)
        H | Qureg2[2]
        CX | (Qureg2[0], Qureg2[1])
        CX | (Qureg2[1], Qureg2[2])
        Y | Qureg2[0]
        S | Qureg2[0]
        Rx(2.0) | Qureg2[1]
        eng2.flush()
        _, v2 = eng2.backend.cheat()
        All(Measure) | Qureg2
        eng2.flush()

        # Compare statevectors
        np.testing.assert_array_equal(v1, v2)

    @unittest.skipIf("projectq" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_projectq2abs(self):
        """ Compares state vector of native ProjectQ circuit against one translated to abstract and then
        ran with projectQ """

        # Compares original abstract circuit to the one obtained by translating to projectQ and back to abstract
        projectq_circ = translator.translate_projectq(abs_circ)
        abs_circ2 = translator._translate_projectq2abs(projectq_circ)
        assert(abs_circ.__str__() == abs_circ2.__str__())

        # Inverse test: assume input is a ProjectQ circuit such as the output of the CommandPrinter engine
        with open(f"{path_data}/projectq_circuit.txt", 'r') as pq_circ_file:
            pq_circ1 = pq_circ_file.read()
            abs_circ1 = translator._translate_projectq2abs(pq_circ1)
            pq_circ2 = translator.translate_projectq(abs_circ1)

            # This package does not generate final measurements and deallocations, so that simulation can retrieve
            # the statevector beforehand. We append them manually for the sake of this test.
            for i in range(abs_circ1.width):
                pq_circ2 += f"Deallocate | Qureg[{i}]\n"
            assert(pq_circ1 == pq_circ2)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_abs2openqasm(self):
        """
            Compares raw openQASM string as generated by Qiskit at the time this test is written, to the one
            this package generates using translate_qiskit and the Qiskit built-in QASM translation

            This test failing implies that either Qiskit QASM output has changed or that translate_qiskit fails
            (the latter has its own tests)
        """
        openqasm_circuit1 = '''OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[2];\ncx q[0],q[1];\ncx q[1],q[2];\ny q[0];\ns q[0];\nrx(1.5) q[1];\nmeasure q[0] -> c[0];\n'''
        openqasm_circuit2 = translator.translate_openqasm(abs_circ_mixed)
        print(openqasm_circuit2)

        # For DEBUG later, if the behavior of Qiskit changes
        # import difflib
        # output_list = [li for li in difflib.ndiff(openqasm_circuit1, openqasm_circuit2) if li[0] != ' ']
        # print(output_list)

        assert(openqasm_circuit1 == openqasm_circuit2)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_openqasm2abs(self):
        """ Translate from abstract format to openQASM and back, compare with original. """
        openqasm_str = translator.translate_openqasm(abs_circ_mixed)
        abs_circ_mixed2 = translator._translate_openqasm2abs(openqasm_str)

        # Two abstract circuits are identical if and only if they have identical string representations
        assert(abs_circ_mixed.__str__() == abs_circ_mixed2.__str__())

    def test_json_ionq(self):
        """ Translate abstract format to IonQ JSON format """

        abs_gates = [Gate("X", 0), Gate("X", 1), Gate("RX", 0, parameter=1.5707963267948966),
                     Gate("H", 2), Gate("CNOT", target=1, control=0), Gate("RZ", 2, parameter=12.566170614359173)]
        abs_circ_ionq = Circuit(abs_gates)
        json_ionq_circ = translator.translate_json_ionq(abs_circ_ionq)

        ref_circuit = {'circuit': [{'gate': 'x', 'target': 0},
                                   {'gate': 'x', 'target': 1},
                                   {'gate': 'rx', 'rotation': 1.5707963267948966, 'target': 0},
                                   {'gate': 'h', 'target': 2},
                                   {'control': 0, 'gate': 'cnot', 'target': 1},
                                   {'gate': 'rz', 'rotation': 12.566170614359173, 'target': 2}],
                       'qubits': 3}

        assert(json_ionq_circ == ref_circuit)

    @unittest.skipIf("braket" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_braket(self):
        """
            Compares the results of a simulation with Braket local simulator using a Braket circuit directly
            VS one obtained through translation from an abstract format
        """
        from braket.circuits import Circuit as BraketCircuit
        from braket.devices import LocalSimulator as BraketLocalSimulator

        # Generate the braket circuit by translating from the abstract one and print it
        translated_circuit = translator.translate_braket(abs_circ)
        print(translated_circuit)

        # Equivalent native braket circuit
        circ = BraketCircuit()
        circ.h(2)
        circ.cnot(0, 1)
        circ.cnot(1, 2)
        circ.y(0)
        circ.s(0)
        circ.rx(1, 2.)
        print(circ)

        # Simulate both circuits on Braket LocalSimulator, assert state vectors are equal
        circ.state_vector()
        translated_circuit.state_vector()

        device = BraketLocalSimulator()
        circ_result = device.run(circ, shots=0).result()
        translated_result = device.run(translated_circuit, shots=0).result()

        np.testing.assert_array_equal(circ_result.values[0], translated_result.values[0])

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_unsupported_gate(self):
        """ Must return an error if a gate is not supported for the target backend """

        circ = Circuit([Gate("Potato", 0)])
        self.assertRaises(ValueError, translator.translate_qiskit, circ)


if __name__ == "__main__":
    unittest.main()
