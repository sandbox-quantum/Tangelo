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
from tangelo.linq import Circuit, Gate
from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq.helpers import pauli_of_to_string, pauli_string_to_of


def get_qiskit_gates():
    """Map gate name of the Tangelo format to the equivalent add_gate method of
    Qiskit's QuantumCircuit class API and supported gates:
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
    """

    import qiskit

    GATE_QISKIT = dict()
    GATE_QISKIT["H"] = qiskit.QuantumCircuit.h
    GATE_QISKIT["X"] = qiskit.QuantumCircuit.x
    GATE_QISKIT["Y"] = qiskit.QuantumCircuit.y
    GATE_QISKIT["Z"] = qiskit.QuantumCircuit.z
    GATE_QISKIT["CH"] = qiskit.QuantumCircuit.ch
    GATE_QISKIT["CX"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["CY"] = qiskit.QuantumCircuit.cy
    GATE_QISKIT["CZ"] = qiskit.QuantumCircuit.cz
    GATE_QISKIT["S"] = qiskit.QuantumCircuit.s
    GATE_QISKIT["T"] = qiskit.QuantumCircuit.t
    GATE_QISKIT["RX"] = qiskit.QuantumCircuit.rx
    GATE_QISKIT["RY"] = qiskit.QuantumCircuit.ry
    GATE_QISKIT["RZ"] = qiskit.QuantumCircuit.rz
    GATE_QISKIT["CRX"] = qiskit.QuantumCircuit.crx
    GATE_QISKIT["CRY"] = qiskit.QuantumCircuit.cry
    GATE_QISKIT["CRZ"] = qiskit.QuantumCircuit.crz
    GATE_QISKIT["CNOT"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["SWAP"] = qiskit.QuantumCircuit.swap
    GATE_QISKIT["XX"] = qiskit.QuantumCircuit.rxx
    GATE_QISKIT["CSWAP"] = qiskit.QuantumCircuit.cswap
    GATE_QISKIT["PHASE"] = qiskit.QuantumCircuit.p
    GATE_QISKIT["CPHASE"] = qiskit.QuantumCircuit.cp
    GATE_QISKIT["MEASURE"] = qiskit.QuantumCircuit.measure
    return GATE_QISKIT


def translate_qiskit(source_circuit):
    """Take in a Circuit, return an equivalent qiskit.QuantumCircuit

    Args:
        source_circuit (Circuit): quantum circuit in the Tangelo format.

    Returns:
        qiskit.QuantumCircuit: the corresponding quantum circuit in Qiskit format.
    """

    return translate_c_to_qiskit(source_circuit)


def translate_c_to_qiskit(source_circuit: Circuit):
    """Take in a Circuit, return an equivalent qiskit.QuantumCircuit

    Args:
        source_circuit (Circuit): quantum circuit in the Tangelo format.

    Returns:
        qiskit.QuantumCircuit: the corresponding qiskit.QuantumCircuit
    """
    import qiskit

    GATE_QISKIT = get_qiskit_gates()

    target_circuit = qiskit.QuantumCircuit(source_circuit.width, source_circuit.width)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.control is not None:
            if len(gate.control) > 1:
                raise ValueError('Multi-controlled gates not supported with qiskit. Gate {gate.name} with controls {gate.control} is not allowed')
        if gate.name in {"H", "Y", "X", "Z", "S", "T"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target[0])
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, gate.target[0])
        elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, gate.control[0], gate.target[0])
        elif gate.name in {"CNOT", "CH", "CX", "CY", "CZ"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.control[0], gate.target[0])
        elif gate.name in {"SWAP"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target[0], gate.target[1])
        elif gate.name in {"CSWAP"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.control[0], gate.target[0], gate.target[1])
        elif gate.name in {"XX"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, gate.target[0], gate.target[1])
        elif gate.name in {"MEASURE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target[0], gate.target[0])
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qiskit")

    return target_circuit


def translate_c_from_qiskit(source_circuit) -> Circuit:
    """Take in a qiskit.QuantumCircuit, return an equivalent Tangelo Circuit

    Args:
        source_circuit (qiskit.QuantumCircuit): quantum circuit in the qiskit.QuantumCircuit format.

    Returns:
        Circuit: the corresponding quantum Circuit in Tangelo format.
    """
    import qiskit

    GATE_QISKIT = get_qiskit_gates()
    qi = source_circuit._qubit_indices
    inv_GATE_QISKIT = {v.__name__: k for k, v in GATE_QISKIT.items()}

    gates = []
    for gate in source_circuit:
        name = inv_GATE_QISKIT[gate.operation.name]
        if name in {"H", "X", "Y", "Z", "S", "T"}:
            gates += [Gate(name, qi[gate.qubits[0]].index)]
        elif name in {"RX", "RY", "RZ", "PHASE"}:
            gates += [Gate(name, qi[gate.qubits[0]].index, parameter=gate.operation.params[0])]
        elif name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            gates += [Gate(name, qi[gate.qubits[1]].index, control=qi[gate.qubits[0]].index, parameter=gate.operation.params[0])]
        elif name in {"CNOT", "CH", "CX", "CY", "CZ"}:
            gates += [Gate(name, qi[gate.qubits[1]].index, control=qi[gate.qubits[0]].index)]
        elif name in {"SWAP"}:
            gates += [Gate(name, [qi[gate.qubits[0]].index, qi[gate.qubits[1]].index])]
        elif name in {"CSWAP"}:
            gates += [Gate(name, [qi[gate.qubits[1]].index, qi[gate.qubits[2]].index], control=qi[gate.qubits[0]].index)]
        elif name in {"XX"}:
            gates += [Gate(name, [qi[gate.qubits[0]].index, qi[gate.qubits[1]].index], parameter=gate.operation.params[0])]
        elif name in {"MEASURE"}:
            gates += [Gate(name, qi[gate.qubits[0]].index)]
        else:
            raise ValueError(f"Gate '{gate.name}' not supported in Tangelo")
    target_circuit = Circuit(gates)

    return target_circuit


def translate_op_to_qiskit(qubit_operator, n_qubits):
    """Helper function to translate a Tangelo QubitOperator to a qiskit
    PauliSumOp. Qiskit must be installed for the function to work.

    Args:
        qubit_operator (tangelo.toolboxes.operators.QubitOperator): Self-explanatory.
        n_qubits (int): Number of qubits relevant to the operator.

    Returns:
        (qiskit.opflow.primitive_ops.PauliSumOp): Qiskit qubit operator.
    """

    # Import qiskit qubit operator.
    from qiskit.opflow.primitive_ops import PauliSumOp

    # Convert each term sequencially.
    term_list = list()
    for term_tuple, coeff in qubit_operator.terms.items():
        term_string = pauli_of_to_string(term_tuple, n_qubits)

        # Reverse the string because of qiskit convention.
        term_list += [(term_string[::-1], coeff)]

    return PauliSumOp.from_list(term_list)


def translate_op_from_qiskit(qubit_operator):
    """Helper function to translate a qiskit PauliSumOp to a Tangelo
    QubitOperator.

    Args:
        qubit_operator (qiskit.opflow.primitive_ops.PauliSumOp): Self-explanatory.

    Returns:
        (tangelo.toolboxes.operators.QubitOperator): Tangelo qubit operator.
    """

    # Create a dictionary to append all terms at once.
    terms_dict = dict()
    for pauli_word in qubit_operator:
        # Inversion of the string because of qiskit ordering.
        term_string = pauli_word.to_pauli_op().primitive.to_label()[::-1]
        term_tuple = pauli_string_to_of(term_string)
        terms_dict[tuple(term_tuple)] = pauli_word.coeff

    # Create and copy the information into a new QubitOperator.
    tangelo_op = QubitOperator()
    tangelo_op.terms = terms_dict

    # Clean the QubitOperator.
    tangelo_op.compress()

    return tangelo_op
