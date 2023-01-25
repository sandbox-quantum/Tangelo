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
format and pennylane format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""

from math import pi

from tangelo.toolboxes.operators import QubitOperator


def get_pennylane_gates():
    """Map gate name of the abstract format to the equivalent methods of the
    pennylane class API and supported gates:https://docs.pennylane.ai/en/stable/code/qml.html#classes.
    """
    import pennylane as qml

    GATE_PENNYLANE = dict()
    GATE_PENNYLANE["H"] = qml.Hadamard
    GATE_PENNYLANE["X"] = qml.PauliX
    GATE_PENNYLANE["Y"] = qml.PauliY
    GATE_PENNYLANE["Z"] = qml.PauliZ
    GATE_PENNYLANE["CX"] = qml.MultiControlledX
    GATE_PENNYLANE["CY"] = qml.CY
    GATE_PENNYLANE["CZ"] = qml.CZ
    GATE_PENNYLANE["S"] = qml.S
    GATE_PENNYLANE["T"] = qml.T
    GATE_PENNYLANE["RX"] = qml.RX
    GATE_PENNYLANE["RY"] = qml.RY
    GATE_PENNYLANE["RZ"] = qml.RZ
    GATE_PENNYLANE["CNOT"] = qml.CNOT
    GATE_PENNYLANE["CRZ"] = qml.CRZ
    GATE_PENNYLANE["CRX"] = qml.CRX
    GATE_PENNYLANE["CRY"] = qml.CRY
    GATE_PENNYLANE["PHASE"] = qml.PhaseShift
    GATE_PENNYLANE["CPHASE"] = qml.CPhase
    GATE_PENNYLANE["XX"] = qml.IsingXX
    GATE_PENNYLANE["SWAP"] = qml.SWAP
    GATE_PENNYLANE["CSWAP"] = qml.CSWAP
    # TODO: Check periodically if better support for measure gates is implemented
    # GATE_PENNYLANE["MEASURE"] = qml.measure  # Pennylane currently only supports measuring a qubit once
    return GATE_PENNYLANE


def translate_c_to_pennylane(source_circuit):
    """Take in an tangelo Circuit, return an equivalent pennylane List[subclass of pennylane.operator.Operation].

    Args:
        source_circuit (Circuit): quantum circuit in the abstract format.

    Returns:
        List[Type[pennylane.operator.Operation]]: corresponding list of objects from child classes of pennylane.operator.Operation.
    """

    GATE_PENNYLANE = get_pennylane_gates()
    target_circuit = []

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit:
        if gate.control is not None and len(gate.control) > 1 and gate.name != "CX":
            raise ValueError(f"Can not use {gate.name} with multiple controls. Only CX translates properly to pennylane")
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            target_circuit.append(GATE_PENNYLANE[gate.name](gate.target[0]))
        elif gate.name in {"CY", "CZ", "CNOT"}:
            target_circuit.append(GATE_PENNYLANE[gate.name]([gate.control[0], gate.target[0]]))
        elif gate.name in {"CX"}:
            target_circuit.append(GATE_PENNYLANE[gate.name](wires=[*gate.control, gate.target[0]]))
        elif gate.name in {"CH"}:
            target_circuit.append(GATE_PENNYLANE["CNOT"]([gate.control[0], gate.target[0]]))
            target_circuit.append(GATE_PENNYLANE["CRY"](-pi/2, [gate.control[0], gate.target[0]]))
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            target_circuit.append(GATE_PENNYLANE[gate.name](gate.parameter, gate.target[0]))
        elif gate.name in {"CRZ", "CRX", "CRY", "CPHASE"}:
            target_circuit.append(GATE_PENNYLANE[gate.name](gate.parameter, [gate.control[0], gate.target[0]]))
        elif gate.name in {"XX"}:
            target_circuit.append(GATE_PENNYLANE[gate.name](gate.parameter, [gate.target[0], gate.target[1]]))
        elif gate.name in {"SWAP"}:
            target_circuit.append(GATE_PENNYLANE[gate.name]([gate.target[0], gate.target[1]]))
        elif gate.name in {"CSWAP"}:
            target_circuit.append(GATE_PENNYLANE[gate.name]([gate.control[0], gate.target[0], gate.target[1]]))
        else:
            raise ValueError(f"Gate '{gate.name}' not supported for translation to pennylane")

    return target_circuit


def translate_op_to_pennylane(qubit_operator):
    """Helper function to translate a Tangelo QubitOperator to a pennylane
    Hamiltonian operator. For the pennylane package, all the coefficients must
    be real.

    Args:
        qubit_operator (tangelo.toolboxes.operators.QubitOperator): Self-explanatory.

    Returns:
        (pennylane.ops.qubit.hamiltonian.Hamiltonian): Pennylane Hamiltonian.
    """
    from pennylane import import_operator

    return import_operator(qubit_operator, format="openfermion")


def translate_op_from_pennylane(qubit_operator):
    """Helper function to translate pennylane Hamiltonian (only real
    coefficients) to a Tangelo QubitOperator.

    Args:
        qubit_operator (pennylane.ops.qubit.hamiltonian.Hamiltonian): Self-explanatory.

    Returns:
        (tangelo.toolboxes.operators.QubitOperator): Tangelo qubit operator.
    """
    from pennylane.qchem.convert import _pennylane_to_openfermion
    of_op = _pennylane_to_openfermion(qubit_operator.coeffs, qubit_operator.ops)

    tangelo_op = QubitOperator()
    tangelo_op.terms = of_op.terms

    return tangelo_op
