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

"""Functions helping with quantum circuit format conversion between abstract
format and ionq format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""

from tangelo.linq import Circuit, Gate


def get_ionq_gates():
    """Map gate name of the abstract format to the equivalent gate name used in
    the json IonQ format. For more information:
    - https://dewdrop.ionq.co/
    - https://docs.ionq.co
    """

    GATE_JSON_IONQ = dict()
    for name in {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "PHASE", "SWAP", "XX"}:
        GATE_JSON_IONQ[name] = name.lower()
    for name in {"CRX", "CRY", "CRZ"}:
        GATE_JSON_IONQ[name] = name[1:].lower()
    for name in {"CX", "CY", "CZ"}:
        GATE_JSON_IONQ[name] = name[1:].lower()
    GATE_JSON_IONQ["CNOT"] = "x"
    GATE_JSON_IONQ["PHASE"] = "z"
    GATE_JSON_IONQ["CPHASE"] = "z"
    return GATE_JSON_IONQ


def translate_c_to_json_ionq(source_circuit):
    """Take in an abstract circuit, return a dictionary following the IonQ JSON
    format as described below.
    https://dewdrop.ionq.co/#json-specification

    Args:
        source_circuit: quantum circuit in the abstract format.

    Returns:
        dict: representation of the quantum circuit following the IonQ JSON
            format.
    """

    GATE_JSON_IONQ = get_ionq_gates()

    json_gates = []
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T", "SWAP"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'targets': gate.target})
        elif gate.name in {"RX", "RY", "RZ", "PHASE", "XX"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'targets': gate.target, 'rotation': gate.parameter})
        elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'targets': gate.target, 'controls': gate.control, 'rotation': gate.parameter})
        elif gate.name in {"CX", "CY", "CZ", "CNOT"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'targets': gate.target, 'controls': gate.control})
        else:
            raise ValueError(f"Gate '{gate.name}' not supported with JSON IonQ translation")

    json_ionq_circ = {"qubits": source_circuit.width, 'circuit': json_gates}
    return json_ionq_circ


def translate_c_from_json_ionq(source_circuit):
    """Take in a dictionary following the IonQ JSON format as described below,
    return an equivalent Tangelo Circuit.
    https://dewdrop.ionq.co/#json-specification

    Args:
        source_circuit (dict): representation of the quantum circuit following
            the IonQ JSON format.

    Returns:
        Circuit: the corresponding quantum Circuit in Tangelo format.
    """

    gates = []
    for gate in source_circuit["circuit"]:
        name = gate["gate"].upper()

        target_qubits = gate.get("target", gate.get("targets"))
        control_qubits = gate.get("control", gate.get("controls"))
        parameter = gate.get("rotation")

        # In Tangelo, the phase gate with a paramter is named PHASE.
        if name == "Z" and parameter is not None:
            name = "PHASE"

        if name in {"H", "X", "Y", "Z", "S", "T", "SWAP"} and control_qubits is None:
            gates += [Gate(name, target_qubits)]
        elif name in {"RX", "RY", "RZ", "PHASE", "XX"} and control_qubits is None:
            gates += [Gate(name, target_qubits, control_qubits, parameter)]
        elif name in {"RX", "RY", "RZ", "PHASE"} and control_qubits is not None:
            gates += [Gate(f"C{name}", target_qubits, control_qubits, parameter)]
        elif name in {"X", "Y", "Z"} and control_qubits is not None and parameter is None:
            gates += [Gate(f"C{name}", target_qubits, control_qubits)]
        else:
            raise ValueError(f"Gate '{name}' not supported in Tangelo")

    target_circuit = Circuit(n_qubits=source_circuit["qubits"]) + Circuit(gates)
    return target_circuit
