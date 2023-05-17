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

"""Functions helping with quantum circuit and operator format conversion between
Tangelo format and qiskit format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.

The module also enables bidirectional conversion between qiskit and Tangelo
qubit operators (linear combination of Pauli operators)
"""

from math import pi, isclose

from tangelo.linq import Circuit, Gate
from tangelo.helpers import deprecated




def get_stim_gates():
    """Map gate name of the Tangelo format to the equivalent add_gate method of
    Stim's CircuitInstruction class API and supported gates:
    https://github.com/quantumlib/Stim/blob/main/doc/gates.md
    """
    GATE_STIM = dict()
    GATE_STIM["I"] = "I"
    GATE_STIM["H"] = "H"
    GATE_STIM["X"] = "X"
    GATE_STIM["Y"] = "Y"
    GATE_STIM["Z"] = "Z"
    GATE_STIM["S"] = "S"
    GATE_STIM["SDAG"] = "S_DAG"
    GATE_STIM["CX"] = "CX"
    GATE_STIM["CY"] = "CY"
    GATE_STIM["CZ"] = "CZ"
    GATE_STIM["CNOT"] = "CNOT"
    GATE_STIM["SWAP"] = "SWAP"
    GATE_STIM["MEASURE"] = "M"
    return GATE_STIM

def decompose_gate_to_cliffords(gate, abs_tol=1e-4):
    """For specific parameters, this function will decompose a single qubit parameterized gate into clifford gates"""

    gate_list = []

    clifford_values = [0, pi, pi / 2, -pi / 2]
    value_isclose = [isclose(gate.parameter, value, abs_tol=abs_tol) for value in clifford_values]

    if not any(value_isclose):
        raise ValueError(
            f"Error: Gate {gate} cannot be decomposed into Clifford gates")
    else:
        clifford_parameter = [value for bool_, value in zip(value_isclose, clifford_values) if bool_][0]

    if clifford_parameter == 0:
        gate_list = [Gate("I", gate.target)]

    elif gate.name == "RY":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("Z", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("Z", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Y", gate.target)]

    elif gate.name == "RX":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("SDAG", gate.target), Gate("H", gate.target), Gate("SDAG", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("SDAG", gate.target), Gate("X", gate.target), Gate("SDAG", gate.target)]

    elif gate.name == "RZ":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("H", gate.target), Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target),
                         Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("SDAG", gate.target), Gate("H", gate.target),
                         Gate("SDAG", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Z", gate.target)]

    elif gate.name == "PHASE":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("SDAG", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("S", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Z", gate.target)]

    return gate_list


def direct_tableau(source_circuit, noise_model=None):
    """Take in an abstract circuit, return an equivalent stim QuantumCircuit
    instance. If provided with a noise model, will add noisy gates at
    translation. Not very useful to look at, as stim does not provide much
    information about the noisy gates added when printing the "noisy circuit".

    Args:
        source_circuit: quantum circuit in the abstract format.
        noise_model: A NoiseModel object from this package, located in the
            # noisy_simulation subpackage.

    Returns:
        stim.QuantumCircuit: the corresponding stim quantum circuit.
    """

    import stim
    from time import time

    GATE_STIM = get_stim_gates()
    target_circuit = stim.TableauSimulator()
    # for qubit in range(source_circuit.width):
    #    target_circuit.

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S"}:
            bar = getattr(target_circuit, gate.name.lower())
            bar(gate.target[0])
            # target_circuit["x"](gate.target[0])
        elif gate.name in {"RY", "RX", "RZ", "PHASE"}:
            # start = time()
            clifford_decomposition = decompose_gate_to_cliffords(gate)

            # print(clifford_decomposition)
            for cliff_gate in clifford_decomposition:
                bar = getattr(target_circuit, GATE_STIM[cliff_gate.name].lower())
                bar(gate.target[0])
                # target_circuit.append(GATE_STIM[cliff_gate.name], [cliff_gate.target[0]])
            # finish = time()
            # print('decomp', finish-start)
        elif gate.name in {"CX", "CY", "CZ", "CNOT", "SWAP"}:
            # target_circuit.append(GATE_STIM[gate.name], [gate.control[0], gate.target[0]])
            bar = getattr(target_circuit, gate.name.lower())
            bar(gate.control[0], gate.target[0])

        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    target_circuit.append(stim.CircuitInstruction('PAULI_CHANNEL_1', [gate.target[0]], [np[0], np[1], np[2]]))
                    if gate.control is not None:
                        target_circuit.append(stim.CircuitInstruction('PAULI_CHANNEL_1', [gate.control[0]], [np[0], np[1], np[2]]))
                if nt == 'depol':
                    if gate.control is not None:
                        target_circuit.depolarize2(zip(gate.target[0], gate.control[0]), np)
                    else:
                        target_circuit.depolarize1([gate.target[0]], [np])

    return target_circuit

@deprecated("Please use the translate_circuit function.")
def translate_stim(source_circuit, noise_model=None):
    """Take in a Circuit, return an equivalent stim.CircuitInstruction

    Args:
        source_circuit (Circuit): quantum circuit in the Tangelo format.

    Returns:
        stim.CircuitInstruction: the corresponding quantum circuit in Qiskit format.
    """
    return translate_c_to_stim(source_circuit, noise_model)

def translate_c_to_stim(source_circuit, noise_model=None):
    """Take in an abstract circuit, return an equivalent stim QuantumCircuit
    instance. If provided with a noise model, will add noisy gates at
    translation. Not very useful to look at, as stim does not provide much
    information about the noisy gates added when printing the "noisy circuit".

    Args:
        source_circuit: quantum circuit in the abstract format.
        noise_model: A NoiseModel object from this package, located in the
            # noisy_simulation subpackage.

    Returns:
        stim.QuantumCircuit: the corresponding stim quantum circuit.
    """

    import stim

    GATE_STIM = get_stim_gates()
    target_circuit = stim.Circuit()
    for qubit in range(source_circuit.width):
        target_circuit.append("I", [qubit])

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S"}:
            target_circuit.append(GATE_STIM[gate.name], [gate.target[0]])
        elif gate.name in {"RY", "RX", "RZ", "PHASE"}:
            clifford_decomposition = decompose_gate_to_cliffords(gate)
            for cliff_gate in clifford_decomposition:
                target_circuit.append(GATE_STIM[cliff_gate.name], [cliff_gate.target[0]])
        elif gate.name in {"CX", "CY", "CZ", "CNOT", "SWAP"}:
            target_circuit.append(GATE_STIM[gate.name], [gate.control[0], gate.target[0]])

        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    target_circuit.append(stim.CircuitInstruction('PAULI_CHANNEL_1', [gate.target[0]], [np[0], np[1], np[2]]))
                    if gate.control is not None:
                        target_circuit.append(stim.CircuitInstruction('PAULI_CHANNEL_1', [gate.control[0]], [np[0], np[1], np[2]]))
                if nt == 'depol':
                    target_circuit.append(
                        stim.CircuitInstruction('DEPOLARIZE1', [gate.target[0]], [np]))
                    if gate.control is not None:
                        target_circuit.append(stim.CircuitInstruction('DEPOLARIZE1', [gate.control[0]], [np]))

    return target_circuit
