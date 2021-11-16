# Copyright 2021 1QB Information Technologies Inc.
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
format and qiskit format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""


def get_qiskit_gates():
    """Map gate name of the abstract format to the equivalent add_gate method of
    Qiskit's QuantumCircuit class API and supported gates:
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
    """

    import qiskit

    GATE_QISKIT = dict()
    GATE_QISKIT["H"] = qiskit.QuantumCircuit.h
    GATE_QISKIT["X"] = qiskit.QuantumCircuit.x
    GATE_QISKIT["Y"] = qiskit.QuantumCircuit.y
    GATE_QISKIT["Z"] = qiskit.QuantumCircuit.z
    GATE_QISKIT["S"] = qiskit.QuantumCircuit.s
    GATE_QISKIT["T"] = qiskit.QuantumCircuit.t
    GATE_QISKIT["RX"] = qiskit.QuantumCircuit.rx
    GATE_QISKIT["RY"] = qiskit.QuantumCircuit.ry
    GATE_QISKIT["RZ"] = qiskit.QuantumCircuit.rz
    GATE_QISKIT["CNOT"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["MEASURE"] = qiskit.QuantumCircuit.measure
    return GATE_QISKIT


def translate_qiskit(source_circuit):
    """Take in an abstract circuit, return an equivalent qiskit QuantumCircuit
    instance

    Args:
        source_circuit: quantum circuit in the abstract format.

    Returns:
        qiskit.QuantumCircuit: the corresponding qiskit quantum circuit.
    """

    import qiskit

    GATE_QISKIT = get_qiskit_gates()
    target_circuit = qiskit.QuantumCircuit(source_circuit.width, source_circuit.width)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, gate.target)
        elif gate.name in {"CNOT"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.control, gate.target)
        elif gate.name in {"MEASURE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target, gate.target)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qiskit")
    return target_circuit
