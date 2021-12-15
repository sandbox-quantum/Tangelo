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

"""Functions helping with quantum circuit format conversion between abstract
format and projectq format

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""

import re
from tangelo.linq import Gate, Circuit


def get_projectq_gates():
    """Map gate name of the abstract format to the equivalent gate name used in
    projectq API and supported gates:
    https://projectq.readthedocs.io/en/latest/projectq.ops.html
    """

    GATE_PROJECTQ = dict()
    for name in {"H", "X", "Y", "Z", "S", "T"}:
        GATE_PROJECTQ[name] = name
    for name in {"RX", "RY", "RZ", "MEASURE"}:
        GATE_PROJECTQ[name] = name[0] + name[1:].lower()
    GATE_PROJECTQ["CNOT"] = "CX"
    GATE_PROJECTQ["PHASE"] = "R"

    return GATE_PROJECTQ


def translate_projectq(source_circuit):
    """Take in an abstract circuit, return a string containing equivalent
    projectq instructions.

    Args:
        source_circuit: quantum circuit in the abstract format.

    Returns:
        str: the corresponding projectq instructions (allocation , gates,
            deallocation).
    """

    GATE_PROJECTQ = get_projectq_gates()

    projectq_circuit = ""
    for i in range(source_circuit.width):
        projectq_circuit += f"Allocate | Qureg[{i}]\n"

    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T", "MEASURE"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]} | Qureg[{gate.target[0]}]\n"
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]}({gate.parameter}) | Qureg[{gate.target[0]}]\n"
        elif gate.name in {"CNOT"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]} | ( Qureg[{gate.control[0]}], Qureg[{gate.target[0]}] )\n"
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend projectQ")

    return projectq_circuit


def _translate_projectq2abs(projectq_str):
    """Convenience function to help people move away from their ProjectQ
    workflow. Take ProjectQ instructions, expressed as a string, similar to one
    from the ProjectQ `CommandPrinter` engine. Will bundle all qubit allocation
    (resp. deallocation) at the beginning (resp. end) of the circuit. Example
    available in the `examples` folder.

    Args:
        projectq_str(str): ProjectQ program, as a string (Allocate, Deallocate,
            gate operations...).

    Returns:
        Circuit: corresponding quantum circuit in the abstract format.
    """

    # Get dictionary of gate mapping, as the reverse dictionary of abs -> projectq translation
    GATE_PROJECTQ = get_projectq_gates()
    gate_mapping = {v: k for k, v in GATE_PROJECTQ.items()}

    # TODO account for mid-circuit measurements, only ignore final measurements
    # Ignore Measure instructions
    projectq_str = re.sub(r'Measure(.*)\n', '', projectq_str)

    # Ignore allocate and deallocate instructions.
    # Number of qubits is inferred by the abstract circuit, no (de)allocation will occur mid-circuit.
    projectq_str = re.sub(r'(.*)llocate(.*)\n', '', projectq_str)
    projectq_gates = [instruction for instruction in projectq_str.split("\n") if instruction]

    # Translate instructions to abstract gates
    abs_circ = Circuit()
    for projectq_gate in projectq_gates:

        # Extract gate name, qubit indices and parameter value (single parameter for now)
        gate_name = re.split(' \| |\(', projectq_gate)[0]
        qubit_indices = [int(index) for index in re.findall('Qureg\[(\d+)\]', projectq_gate)]
        parameters = [float(index) for index in re.findall('\((.*)\)', projectq_gate) if "Qureg" not in index]

        if gate_name in {"H", "X", "Y", "Z", "S", "T"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[0])
        elif gate_name in {"Rx", "Ry", "Rz", "PHASE"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[0], parameter=parameters[0])
        # #TODO: Rethink the use of enums for gates to set the equality CX=CNOT and enable other refactoring
        elif gate_name in {"CX"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[1], control=qubit_indices[0])
        else:
            raise ValueError(f"Gate '{gate_name}' not supported with project2abs translation")
        abs_circ.add_gate(gate)

    return abs_circ
