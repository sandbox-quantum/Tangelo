"""Module to create Hardware Efficient Ansatze (HEA) circuit with n_layers"""

from agnostic_simulator import Circuit, Gate


def EulerCircuit(n_qubits):
    """Construct a circuit applying an Euler Z-X-Z rotation to each qubit."""
    circuit = Circuit()
    for target in range(n_qubits):
        circuit.add_gate(Gate("RZ" , target, parameter=0.0, is_variational=True))
        circuit.add_gate(Gate("RX", target, parameter=0.0, is_variational=True))
        circuit.add_gate(Gate("RZ", target, parameter=0.0, is_variational=True))
    return circuit


def EntanglerCircuit(n_qubits):
    """Construct a circuit applying two columns of staggered CNOT gates to all qubits
     and their neighbours"""
    circuit = Circuit()
    for ii in range(n_qubits//2):
        circuit.add_gate(Gate("CNOT", control=2*ii, target=2*ii + 1))
    for ii in range(n_qubits//2 - 1):
        circuit.add_gate(Gate("CNOT", control=2*ii + 1, target=2*(ii+1)))
    return circuit


def HEACircuit(n_qubits, n_layers):
    """Construct a circuit consisting of alternating sequence of Euler rotations and entanglers"""
    circuit = EulerCircuit(n_qubits)
    for ii in range(n_layers):
        circuit += EntanglerCircuit(n_qubits)
        circuit += EulerCircuit(n_qubits)
    return circuit
