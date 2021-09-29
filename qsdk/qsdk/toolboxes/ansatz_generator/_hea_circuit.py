"""Module to create Hardware Efficient Ansatze (HEA) circuit with n_layers."""

from agnostic_simulator import Circuit, Gate


def rotation_circuit(n_qubits, rot_type="euler"):
    """Construct a circuit applying an Euler Z-X-Z rotation to each qubit."""
    if rot_type == "euler":
        gateslist = [Gate("RZ", target=target, is_variational=True) for target in range(n_qubits)]
        gateslist.extend(Gate("RX", target=target, is_variational=True) for target in range(n_qubits))
        gateslist.extend(Gate("RZ", target=target, is_variational=True) for target in range(n_qubits))
    elif rot_type == "real":
        gateslist = [Gate("RY", target, is_variational=True) for target in range(n_qubits)]
    else:
        raise ValueError(f"Supported keywords for initializing variational parameters: ['real', 'euler'']")
    return Circuit(gateslist)


def entangler_circuit(n_qubits):
    """Construct a circuit applying two columns of staggered CNOT gates to all
    qubits and their neighbours.
    """

    gateslist = [Gate("CNOT", control=2*ii, target=2*ii + 1) for ii in range(n_qubits//2)]
    gateslist.extend(Gate("CNOT", control=2*ii + 1, target=2*(ii+1)) for ii in range(n_qubits//2 - 1))

    return Circuit(gateslist)


def construct_hea_circuit(n_qubits, n_layers, rot_type="euler"):
    """Construct a circuit consisting of alternating sequence of Euler rotations
    and entanglers.
    """
    circuit = rotation_circuit(n_qubits, rot_type)
    for ii in range(n_layers):
        circuit += entangler_circuit(n_qubits)
        circuit += rotation_circuit(n_qubits, rot_type)
    return circuit
