import copy
import numpy as np

from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq import Circuit
from tangelo.linq.helpers.circuits import pauli_string_to_of, pauli_of_to_string


def trim_trivial_operator(qu_op, n_qubits, trim_index, trim_states, reindex=True):
    """
    Calculates expectation values of a QubitOperator acting on qubits in a
    trivial |0> or |1> state. Returns a trimmed QubitOperator with updated coefficients

    Args:
        qu_op (QubitOperator): Operator to trim
        n_qubits (int): number of qubits in full system
        trim_index (list):  index of qubits to trim
        trim_states (list): state of the qubits to trim, 0 or 1
        reindex (bool): If true, remaining qubits will be reindexed
    Returns:
        qu_op  (QubitOperator): trimmed QubitOperator with updated coefficients
    """
    qu_op_trim = QubitOperator()
    trim_states = [x for (y, x) in sorted(zip(trim_index, trim_states), key=lambda pair: pair[0])]
    trim_index = sorted(trim_index)

    # Calculate expectation values of trivial qubits, update coefficients
    for op, coeff in qu_op.terms.items():
        term = pauli_of_to_string(op, n_qubits)
        c = np.ones(len(trim_index))
        new_term = copy.copy(term)
        for i, qubit in enumerate(trim_index):
            if term[qubit] == 'X' or term[qubit] == 'Y':
                c[i] = 0
            elif term[qubit] == 'Z':
                if trim_states[i] == 0:
                    c[i] = 1
                elif trim_states[i] == 1:
                    c[i] = -1
            # Reindex Hamiltonian
            if reindex is True:
                new_term = new_term[:qubit-i] + new_term[qubit-i + 1:]
            else:
                new_term = new_term[:qubit] + 'I' + new_term[qubit + 1:]

        qu_op_trim += np.prod(c)*coeff*QubitOperator(pauli_string_to_of(new_term))

    return qu_op_trim


def trim_trivial_circuit(circuit):
    """
        Splits Circuit into entangled. and unentangled components.
        Returns entangled Circuit, and the indices and states of unentangled qubits

        Args:
            circuit (Circuit): circuit to be trimmed
        Returns:
            circuit_new (Circuit) : Trimmed, entangled circuit
            trim_states (list): state of removed qubits, 0 or 1
            trim_index (list):  index of removed qubits

    """
    # Simplify circuit
    circuit.remove_redundant_gates()
    circuit.remove_small_rotations()

    # Split circuit into components with entangling and nonentangling gates
    circs = circuit.split()
    e_indices = circuit.get_entangled_indices()

    # Get list of qubits with gates applied to them
    gated_qubits = []
    for element in e_indices:
        gated_qubits += element

    # Find qubits with no gates applied to them, store qubit index and state |0>
    zeros = list(sorted(set(range(circuit.width)) - set(gated_qubits)))
    trim_index = [i for i in zeros]
    trim_states = [0 for i in range(len(zeros))]

    # Go through circuit components, trim if trivial, otherwise append to new circuit
    circuit_new = Circuit()
    for i, circ in enumerate(circs):
        if circ.width == 1:
            if len(circ._gates) == 1 and circ._gates[0].name == 'X':
                trim_index.append(e_indices[i].pop())
                trim_states.append(1)
            else:
                circuit_new += circ
        else:
            circuit_new += circ

    return circuit_new, trim_index, trim_states


def trim_trivial_qubits(operator, circuit):
    """
        Trims circuit and operator based on expectation values calculated from
        trivial components of the circuit.

        Args:
            circuit (Circuit): circuit to be trimmed
            operator (QubitOperator): Operator to trim
        Returns:
            trimmed_circuit (Circuit): circuit to be trimmed
            trimmed_operator (QubitOperator): Operator to trim

    """
    trimmed_circuit, trim_index, trim_states = trim_trivial_circuit(circuit)
    trimmed_operator = trim_trivial_operator(operator, circuit.width, trim_index, trim_states, reindex=True)

    return trimmed_circuit, trimmed_operator
