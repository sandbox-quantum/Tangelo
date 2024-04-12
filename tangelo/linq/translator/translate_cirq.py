# Copyright SandboxAQ 2021-2024.
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

"""Functions helping with quantum circuit format conversion between abstract
format and circ format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""

from math import pi

from tangelo.toolboxes.operators import QubitOperator


def get_cirq_gates():
    """Map gate name of the abstract format to the equivalent methods of the
    cirq class API and supported gates: https://quantumai.google/cirq/gates.
    """
    import cirq

    GATE_CIRQ = dict()
    GATE_CIRQ["H"] = cirq.H
    GATE_CIRQ["X"] = cirq.X
    GATE_CIRQ["Y"] = cirq.Y
    GATE_CIRQ["Z"] = cirq.Z
    GATE_CIRQ["CX"] = cirq.X
    GATE_CIRQ["CY"] = cirq.Y
    GATE_CIRQ["CZ"] = cirq.Z
    GATE_CIRQ["S"] = cirq.S
    GATE_CIRQ["SDAG"] = cirq.ZPowGate(exponent=-0.5)
    GATE_CIRQ["T"] = cirq.T
    GATE_CIRQ["CH"] = cirq.H
    GATE_CIRQ["RX"] = cirq.rx
    GATE_CIRQ["RY"] = cirq.ry
    GATE_CIRQ["RZ"] = cirq.rz
    GATE_CIRQ["CNOT"] = cirq.CNOT
    GATE_CIRQ["CRZ"] = cirq.rz
    GATE_CIRQ["CRX"] = cirq.rx
    GATE_CIRQ["CRY"] = cirq.ry
    GATE_CIRQ["PHASE"] = cirq.ZPowGate
    GATE_CIRQ["CPHASE"] = cirq.ZPowGate
    GATE_CIRQ["XX"] = cirq.XXPowGate
    GATE_CIRQ["SWAP"] = cirq.SWAP
    GATE_CIRQ["CSWAP"] = cirq.SWAP
    GATE_CIRQ["MEASURE"] = cirq.measure
    GATE_CIRQ["CMEASURE"] = cirq.measure
    return GATE_CIRQ


def translate_c_to_cirq(source_circuit, noise_model=None, save_measurements=False):
    """Take in an abstract circuit, return an equivalent cirq QuantumCircuit
    object.

    Args:
        source_circuit (Circuit): quantum circuit in the abstract format.
        noise_model (NoiseModel): The noise model to use
        save_measurements (bool): If True, all measurements in the circuit are saved
            with the key 'n' for the nth measurement in the Circuit. If False, no
            measurements are saved.

    Returns:
        cirq.Circuit: a corresponding cirq Circuit. Right now, the structure is
            of LineQubit. It is possible in the future that we may support
            NamedQubit or GridQubit.
    """
    import cirq

    GATE_CIRQ = get_cirq_gates()
    target_circuit = cirq.Circuit()
    # cirq by definition uses labels for qubits, this is one way to automatically generate
    # labels. Could also use GridQubit for square lattice or NamedQubit to name qubits
    qubit_list = cirq.LineQubit.range(source_circuit.width)
    # Add next line to make sure all qubits are initialized
    # cirq will otherwise only initialize qubits that have gates
    target_circuit.append(cirq.I.on_each(qubit_list))

    measure_count = 0

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.control is not None:
            num_controls = len(gate.control)
            control_list = [qubit_list[c] for c in gate.control]
            if gate.name == 'CNOT' and num_controls > 1:
                gate.name = 'CX'
        if gate.name in {"H", "X", "Y", "Z", "S", "SDAG", "T"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.target[0]]))
        elif gate.name in {"CH", "CX", "CY", "CZ"}:
            next_gate = GATE_CIRQ[gate.name].controlled(num_controls)
            target_circuit.append(next_gate(*control_list, qubit_list[gate.target[0]]))
        elif gate.name in {"RX", "RY", "RZ"}:
            next_gate = GATE_CIRQ[gate.name](gate.parameter)
            target_circuit.append(next_gate(qubit_list[gate.target[0]]))
        elif gate.name in {"CNOT"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.control[0]], qubit_list[gate.target[0]]))
        elif gate.name in {"MEASURE"}:
            key = str(measure_count) if save_measurements else None
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.target[0]], key=key))
            measure_count += 1
        elif gate.name in {"CRZ", "CRX", "CRY"}:
            next_gate = GATE_CIRQ[gate.name](gate.parameter).controlled(num_controls)
            target_circuit.append(next_gate(*control_list, qubit_list[gate.target[0]]))
        elif gate.name in {"XX"}:
            next_gate = GATE_CIRQ[gate.name](exponent=gate.parameter/pi, global_shift=-0.5)
            target_circuit.append(next_gate(qubit_list[gate.target[0]], qubit_list[gate.target[1]]))
        elif gate.name in {"PHASE"}:
            next_gate = GATE_CIRQ[gate.name](exponent=gate.parameter/pi)
            target_circuit.append(next_gate(qubit_list[gate.target[0]]))
        elif gate.name in {"CPHASE"}:
            next_gate = GATE_CIRQ[gate.name](exponent=gate.parameter/pi).controlled(num_controls)
            target_circuit.append(next_gate(*control_list, qubit_list[gate.target[0]]))
        elif gate.name in {"SWAP"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.target[0]], qubit_list[gate.target[1]]))
        elif gate.name in {"CSWAP"}:
            next_gate = GATE_CIRQ[gate.name].controlled(num_controls)
            target_circuit.append(next_gate(*control_list, qubit_list[gate.target[0]], qubit_list[gate.target[1]]))
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend cirq")

        # Add noisy gates
        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    # Define pauli gate in cirq language
                    depo = cirq.asymmetric_depolarize(np[0], np[1], np[2])

                    target_circuit += [depo(qubit_list[t]) for t in gate.target]
                    if gate.control is not None:
                        target_circuit += [depo(qubit_list[c]) for c in gate.control]
                elif nt == 'depol':
                    depo_list = [qubit_list[t] for t in gate.target]
                    if gate.control is not None:
                        depo_list += [qubit_list[c] for c in gate.control]
                    depo_size = len(depo_list)
                    # define depo_size-qubit depolarization gate
                    depo = cirq.depolarize(np*(4**depo_size-1)/4**depo_size, depo_size)  # param, num_qubits
                    target_circuit.append(depo(*depo_list))  # gates targeted

    return target_circuit


def translate_op_to_cirq(qubit_operator):
    """Helper function to translate a Tangelo QubitOperator to a cirq linear
    combination of Pauli strings.

    Args:
        qubit_operator (tangelo.toolboxes.operators.QubitOperator): Self-explanatory.

    Returns:
        (cirq.ops.linear_combinations.PauliSum): Cirq summation of Pauli strings.
    """
    from openfermion.transforms import qubit_operator_to_pauli_sum

    return qubit_operator_to_pauli_sum(qubit_operator)


def translate_op_from_cirq(qubit_operator):
    """Helper function to translate a cirq linear combination of Pauli strings
    to a Tangelo QubitOperator.

    Args:
        qubit_operator (cirq.ops.linear_combinations.PauliSum): Self-explanatory.

    Returns:
        (tangelo.toolboxes.operators.QubitOperator): Tangelo qubit operator.
    """
    import cirq

    tangelo_op = QubitOperator()

    cirq_pauli_to_string = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}

    for pauli_word in qubit_operator:

        term_string = ""
        for line_qubit, cirq_pauli in pauli_word.items():
            term_string += f"{cirq_pauli_to_string[cirq_pauli]}{line_qubit.x} "

        tangelo_op += QubitOperator(term_string.strip(), pauli_word.coefficient)

    return tangelo_op
