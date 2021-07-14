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


def circuit_from_exponentiated_qubit_op(qubit_op, time=1., variational=False):
    """Generates the exponentiation of a qubit operator in first_order Trotterized form
       as a quantum circuit as described in Whitfield 2010 https://arxiv.org/pdf/1001.3855.pdf

    Args: qubit_op  (QubitOperator):  qubit hamiltonian to exponentiate
          time (float or dict): with time or value of exponential for each term
          variational=False whether the coefficients are variational

    Returns: Circuit corresponding to exponentiation of qubit operator"""

    pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))

    len_qubit_op = len(qubit_op.terms)
    if isinstance(time, float):
        for k, v in pauli_words:
            pauli_words[k] = time*v
    elif isinstance(time, dict):
        try:
            assert (len(time) == len_qubit_op)
            for k, v in pauli_words:
                if k in time:
                    pauli_words[k] = time[k]*v
                else:
                    ValueError(f'no time parameter given for pauli_word {k} in time dictionary')
        except AssertionError:
            raise ValueError(f"Expected {len_qubit_op} time parameters but received {len(time)}.")
    else:
        raise TypeError(f'Expected Float or np.array and received {time} with {type(time)}')

    pauli_words_gates = []
    for pauli_word, coef in pauli_words:
        pauli_words_gates += pauliword_to_circuit(pauli_word, coef, variational=variational)

    return Circuit(pauli_words_gates)


def pauliword_to_circuit(pauli_word, coef, variational=True):
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
    gates += [Gate("RZ", target=indices[-1], parameter=angle, is_variational=variational)]

    gates += cnot_ladder_gates[::-1]

    # After CNOT ladder
    for index, op in pauli_word[::-1]:
        if op in {"X", "Y"}:
            gates += [pauli_op_to_gate(index, op, inverse=True)]

    return gates
