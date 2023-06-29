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
format and SYMPY format

In order to produce an equivalent circuit for the target backend, it is necessary
to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""

from tangelo.linq.helpers import pauli_of_to_string


def rx_gate(target, theta):
    """Implementation of the RX gate with a unitary matrix.

    Args:
        target (int): Target qubit.
        theta (float): Rotation angle.

    Returns:
        sympy.physics.quantum.gate.UGate: Self-explanatory.
    """

    from sympy import ImmutableMatrix, sin, cos, I
    from sympy.physics.quantum.gate import UGate

    cos_term = cos(theta / 2)
    sin_term = -I*sin(theta / 2)
    rx_matrix = ImmutableMatrix([[cos_term, sin_term], [sin_term, cos_term]])

    return UGate(target, rx_matrix)


def ry_gate(target, theta):
    """Implementation of the RY gate with a unitary matrix.

    Args:
        target (int): Target qubit.
        theta (float): Rotation angle.

    Returns:
        sympy.physics.quantum.gate.UGate: Self-explanatory.
    """

    from sympy import ImmutableMatrix, sin, cos
    from sympy.physics.quantum.gate import UGate

    cos_term = cos(theta / 2)
    sin_term = sin(theta / 2)
    ry_matrix = ImmutableMatrix([[cos_term, -sin_term], [sin_term, cos_term]])

    return UGate(target, ry_matrix)


def rz_gate(target, theta):
    """Implementation of the RZ gate with a unitary matrix.

    Args:
        target (int): Target qubit.
        theta (float): Rotation angle.

    Returns:
        sympy.physics.quantum.gate.UGate: Self-explanatory.
    """

    from sympy import ImmutableMatrix, I, exp
    from sympy.physics.quantum.gate import UGate

    rz_matrix = ImmutableMatrix([[exp(- I * theta / 2), 0], [0, exp(I * theta / 2)]])

    return UGate(target, rz_matrix)


def p_gate(target, theta):
    """Implementation of the PHASE gate with a unitary matrix.

    Args:
        target (int): Target qubit.
        theta (float): Phase parameter.

    Returns:
        sympy.physics.quantum.gate.UGate: Self-explanatory.
    """

    from sympy import ImmutableMatrix, I, exp
    from sympy.physics.quantum.gate import UGate

    p_matrix = ImmutableMatrix([[1, 0], [0, exp(I * theta)]])

    return UGate(target, p_matrix)


def controlled_gate(gate_function):
    """Change a one-qubit gate to a controlled gate.

    Args:
        gate_function (sympy.physics.quantum.gate): Class for a specific gate.

    Returns:
        function: Function wrapping a gate with CGate.
    """

    from sympy.physics.quantum.gate import CGate

    def cgate(control, target, *args, **kwargs):
        return CGate(control, gate_function(target, *args, **kwargs))

    return cgate


def get_sympy_gates():
    """Map gate name of the abstract format to the equivalent methods of the
    sympy backend (symbolic simulation).
    """

    import sympy.physics.quantum.gate as SYMPYGate

    GATE_SYMPY = dict()
    GATE_SYMPY["H"] = SYMPYGate.HadamardGate
    GATE_SYMPY["X"] = SYMPYGate.XGate
    GATE_SYMPY["Y"] = SYMPYGate.YGate
    GATE_SYMPY["Z"] = SYMPYGate.ZGate
    GATE_SYMPY["S"] = SYMPYGate.PhaseGate
    GATE_SYMPY["T"] = SYMPYGate.TGate
    GATE_SYMPY["PHASE"] = p_gate
    GATE_SYMPY["SWAP"] = SYMPYGate.SwapGate
    GATE_SYMPY["RX"] = rx_gate
    GATE_SYMPY["RY"] = ry_gate
    GATE_SYMPY["RZ"] = rz_gate
    GATE_SYMPY["RX"] = rx_gate
    GATE_SYMPY["CH"] = controlled_gate(SYMPYGate.HadamardGate)
    GATE_SYMPY["CNOT"] = SYMPYGate.CNotGate
    GATE_SYMPY["CX"] = SYMPYGate.CNotGate
    GATE_SYMPY["CY"] = controlled_gate(SYMPYGate.YGate)
    GATE_SYMPY["CZ"] = controlled_gate(SYMPYGate.ZGate)
    GATE_SYMPY["CRX"] = controlled_gate(rx_gate)
    GATE_SYMPY["CRY"] = controlled_gate(ry_gate)
    GATE_SYMPY["CRZ"] = controlled_gate(rz_gate)
    GATE_SYMPY["CS"] = controlled_gate(SYMPYGate.PhaseGate)
    GATE_SYMPY["CT"] = controlled_gate(SYMPYGate.TGate)
    GATE_SYMPY["CPHASE"] = controlled_gate(p_gate)

    return GATE_SYMPY


def translate_c_to_sympy(source_circuit):
    """Take in an abstract circuit, return a quantum circuit object as defined
    in the Python SYMPY SDK.

    Args:
        source_circuit: quantum circuit in the abstract format.

    Returns:
        SYMPY.circuits.Circuit: quantum circuit in Python SYMPY SDK format.
    """

    from sympy import symbols

    GATE_SYMPY = get_sympy_gates()

    # Identity as an empty circuit.
    target_circuit = 1

    # Map the gate information properly.
    for gate in reversed(source_circuit._gates):
        # If the parameter is a string, we use it as a variable.
        if gate.parameter and isinstance(gate.parameter, str):
            gate.parameter = symbols(gate.parameter, real=True)

        if gate.name in {"H", "X", "Y", "Z"}:
            target_circuit *= GATE_SYMPY[gate.name](gate.target[0])
        elif gate.name in {"T", "S"} and gate.parameter == "":
            target_circuit *= GATE_SYMPY[gate.name](gate.target[0])
        elif gate.name in {"PHASE", "RX", "RY", "RZ"}:
            target_circuit *= GATE_SYMPY[gate.name](gate.target[0], gate.parameter)
        elif gate.name in {"CNOT", "CH", "CX", "CY", "CZ", "CS", "CT"}:
            target_circuit *= GATE_SYMPY[gate.name](gate.control[0], gate.target[0])
        elif gate.name in {"SWAP"}:
            target_circuit *= GATE_SYMPY[gate.name](gate.target[0], gate.target[1])
        elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            target_circuit *= GATE_SYMPY[gate.name](gate.control[0], gate.target[0], gate.parameter)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend SYMPY")

    return target_circuit


def translate_op_to_sympy(qubit_operator, n_qubits):
    """Helper function to translate a Tangelo QubitOperator to a sympy linear
    combination of tensor products.

    Args:
        qubit_operator (tangelo.toolboxes.operators.QubitOperator): Self-explanatory.
        n_qubits (int): The number of qubit the operator acts on.

    Returns:
        sympy.core.add.Add: Summation of sympy.physics.quantum.TensorProduct
            objects.
    """
    from sympy import Matrix, I, zeros
    from sympy.physics.quantum import TensorProduct

    # Pauli string to sympy Pauli algebra objects.
    map_to_paulis = {
        "I": Matrix([[1, 0], [0, 1]]),
        "X": Matrix([[0, 1], [1, 0]]),
        "Y": Matrix([[0, -I], [I, 0]]),
        "Z": Matrix([[1, 0], [0, -1]])
    }

    # Contruct the TensorProduct objects.
    sum_tensor_paulis = zeros(2**n_qubits, 2**n_qubits)
    for term_tuple, coeff in qubit_operator.terms.items():
        term_string = pauli_of_to_string(term_tuple, n_qubits)
        paulis = [map_to_paulis[p] for p in term_string[::-1]]
        tensor_paulis = TensorProduct(*paulis)
        sum_tensor_paulis += coeff * tensor_paulis

    return sum_tensor_paulis
