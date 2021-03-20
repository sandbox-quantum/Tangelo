"""
    Provide useful functions, corresponding to common patterns in quantum chemistry circuits (CNOT ladders,
    Pauli-word to circuit translation ...) to facilitate the assembly of ansatz quantum circuits
"""

import numpy as np
from agnostic_simulator import Circuit, Gate


def pauli_op_to_gate(index, op, inverse=False):
    """ Return the change-of-basis gates required to map pauli words to quantum circuit as per Whitfield 2010
        https://arxiv.org/pdf/1001.3855.pdf
    """
    if op == 'X':
        return Gate("H", index)
    elif op == 'Y':
        angle = 0.5*np.pi
        return Gate("RX", index, parameter=angle) if not inverse else Gate("RX", index, parameter=-angle+4*np.pi)


def pauliword_to_circuit(pauli_word, coef):
    """ Generates a quantum circuit corresponding to the pauli word, as described in Whitfield 2010
        https://arxiv.org/pdf/1001.3855.pdf
    """
    gates = []

    # Before CNOT ladder
    for index, op in pauli_word:
        if op in {"X", "Y"}:
            gates += [pauli_op_to_gate(index, op, inverse=False)]

    # CNOT ladder and rotation
    indices = sorted([index for index, op in pauli_word])
    cnot_ladder_gates = [Gate("CNOT", target=pair[1], control=pair[0]) for pair in zip(indices[:-1], indices[1:])]
    gates += cnot_ladder_gates

    angle = 2.*coef if coef >= 0. else 4*np.pi+2*coef
    gates += [Gate("RZ", target=indices[-1], parameter=angle, is_variational=True)]

    gates += cnot_ladder_gates[::-1]

    # After CNOT ladder
    for index, op in pauli_word[::-1]:
        if op in {"X", "Y"}:
            gates += [pauli_op_to_gate(index, op, inverse=True)]

    return gates
