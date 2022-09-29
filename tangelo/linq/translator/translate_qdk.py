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
format and qdk/qsharp format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""


def get_qdk_gates():
    """Map gate name of the abstract format to the equivalent gate name used in
    Q# operations API and supported gates:
    https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic
    """

    GATE_QDK = dict()
    for name in {"H", "X", "Y", "Z", "S", "T", "CNOT"}:
        GATE_QDK[name] = name
    for name in {"RX", "RY", "RZ"}:
        GATE_QDK[name] = name[0] + name[1:].lower()
    GATE_QDK["PHASE"] = "R1"
    GATE_QDK["CPHASE"] = "R1"
    for name in {"CRX", "CRY", "CRZ"}:
        GATE_QDK[name] = name[1] + name[2:].lower()
    for name in {"CH", "CX", "CY", "CZ", "CS", "CT", "CSWAP"}:
        GATE_QDK[name] = name[1:]
    GATE_QDK["SWAP"] = "SWAP"
    GATE_QDK["MEASURE"] = "M"

    return GATE_QDK


def translate_qsharp(source_circuit, operation="MyQsharpOperation"):
    """Take in an abstract circuit, generate the corresponding Q# operation
    (state prep + measurement) string, in the appropriate Q# template. The Q#
    output can be written to file and will be compiled at runtime.

    Args:
        source_circuit: quantum circuit in the abstract format.
        operation (str), optional: name of the Q# operation.

    Returns:
        str: The Q# code (operation + template). This needs to be written into a
            .qs file, and compiled at runtime.
    """

    GATE_QDK = get_qdk_gates()

    # Prepare Q# operation header
    qsharp_string = ""
    qsharp_string += "@EntryPoint()\n"
    qsharp_string += f"operation {operation}() : Result[] {{\n"
#    qsharp_string += "body (...) {\n\n"
    qsharp_string += f"\tmutable c = new Result[{source_circuit.width}];\n"
    qsharp_string += f"\tusing (qreg = Qubit[{source_circuit.width}]) {{\n"

    # Generate Q# strings with the right syntax, order and values for the gate inputs
    body_str = ""
    for gate in source_circuit._gates:
        if gate.control is not None and gate.name != "CNOT":
            control_string = '['
            num_controls = len(gate.control)
            for i, c in enumerate(gate.control):
                control_string += f'qreg[{c}]]' if i == num_controls - 1 else f'qreg[{c}], '

        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}(qreg[{gate.target[0]}]);\n"
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}({gate.parameter}, qreg[{gate.target[0]}]);\n"
        elif gate.name in {"CNOT"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}(qreg[{gate.control[0]}], qreg[{gate.target[0]}]);\n"
        elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            body_str += f"\t\tControlled {GATE_QDK[gate.name]}({control_string}, ({gate.parameter}, qreg[{gate.target[0]}]));\n"
        elif gate.name in {"CH", "CX", "CY", "CZ", "CS", "CT"}:
            body_str += f"\t\tControlled {GATE_QDK[gate.name]}({control_string}, (qreg[{gate.target[0]}]));\n"
        elif gate.name in {"SWAP"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}(qreg[{gate.target[0]}], qreg[{gate.target[1]}]);\n"
        elif gate.name in {"CSWAP"}:
            body_str += f"\t\tControlled {GATE_QDK[gate.name]}({control_string}, (qreg[{gate.target[0]}], qreg[{gate.target[1]}]));\n"
        elif gate.name in {"MEASURE"}:
            body_str += f"\t\tset c w/= {gate.target[0]} <- {GATE_QDK[gate.name]}(qreg[{gate.target[0]}]);\n"
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qdk")
    qsharp_string += body_str + "\n\t\treturn ForEach(MResetZ, qreg);\n"
    qsharp_string += "\t}\n"
#    qsharp_string += "}\n adjoint auto;\n"
    qsharp_string += "}\n"

    # Fills the template with the Q# operation. The written Q# file will be compiled by the qsharp module at runtime
    from .qdk_template import _qdk_template, _header
    template = _header + _qdk_template.format(operation_name=operation)

    # Return Q# code (template + operation) for simulation. Needs to be written to file and compiled at runtime
    return template + qsharp_string + "\n}\n"
