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

import numpy as np

from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq import Circuit
from tangelo.linq.helpers.circuits import pauli_string_to_of, pauli_of_to_string
from tangelo.toolboxes.operators import count_qubits


def trim_trivial_operator(qu_op, trim_index, trim_states, n_qubits=None, reindex=True):
    """
    Calculate expectation values of a QubitOperator acting on qubits in a
    trivial |0> or |1> state. Return a trimmed QubitOperator with updated coefficients

    Args:
        qu_op (QubitOperator): Operator to trim
        trim_index (list):  index of qubits to trim
        trim_states (list): state of the qubits to trim, 0 or 1
        n_qubits (int): Optional, number of qubits in full system
        reindex (bool): Optional, if True, remaining qubits will be reindexed
    Returns:
        QubitOperator : trimmed QubitOperator with updated coefficients
    """
    trim_states = [x for (y, x) in sorted(zip(trim_index, trim_states), key=lambda pair: pair[0])]
    trim_index = sorted(trim_index)
    qu_op_trim = QubitOperator()
    n_qubits = count_qubits(qu_op) if n_qubits is None else n_qubits

    # Calculate expectation values of trivial qubits, update coefficients
    for op, coeff in qu_op.terms.items():
        term = pauli_of_to_string(op, n_qubits)
        c = np.ones(len(trim_index))
        new_term = term
        for i, qubit in enumerate(trim_index):
            if term[qubit] in {'X', 'Y'}:
                c[i] = 0
            elif (term[qubit], trim_states[i]) == ('Z', 1):
                c[i] = -1

            new_term = new_term[:qubit - i] + new_term[qubit - i + 1:] if reindex else new_term[:qubit] + 'I' + new_term[qubit + 1:]

        qu_op_trim += np.prod(c) * coeff * QubitOperator(pauli_string_to_of(new_term))
    return qu_op_trim

def is_bitflip_gate(gate):
    """
    Check if a gate is a bitflip gate.

    A gate is a bitflip gate if it satisfies one of the following conditions:
    1. The gate name is either X, Y.
    2. The gate has a parameter that is an odd multiple of pi.

    Args:
        gate (Gate): The gate to check.

    Returns:
        bool: True if the gate is a single qubit bitflip gate, False otherwise.
    """
    if gate is None:
        return False

    if gate.name in {"X", "Y"}:
        return True
    elif gate.name in {"RX", "RY"}:
        try:
            parameter_float = float(gate.parameter)
        except (TypeError, ValueError):
            return False

        return abs(parameter_float - round(parameter_float / np.pi) * np.pi) < 1e-5 and round(parameter_float / np.pi) % 2 == 1
    else:
        return False

def trim_trivial_circuit(circuit):
    """
        Splits Circuit into entangled and unentangled components.
        Returns entangled Circuit, and the indices and states of unentangled qubits

        Args:
            circuit (Circuit): circuit to be trimmed
        Returns:
            Circuit : Trimmed, entangled circuit
            list : state of removed qubits, 0 or 1
            list :  indices of removed qubits

    """
    # Split circuit into components with entangling and nonentangling gates
    circs = circuit.split()
    e_indices = circuit.get_entangled_indices()

    # Get list of qubits with gates applied to them
    gated_qubits = [qubit for e_subset in e_indices for qubit in e_subset]

    # Find qubits with no gates applied to them, store qubit index and state |0>
    trim_index = list(sorted(set(range(circuit.width)) - set(gated_qubits)))
    trim_states = [0]*len(trim_index)

    circuit_new = Circuit()
    # Go through circuit components, trim if trivial, otherwise append to new circuit
    for i, circ in enumerate(circs):
        if circ.width != 1:
            circuit_new += circ
            continue
        # Calculate state of single qubit clifford circuits, ideally this would be done with a clifford simulator
        # for now only look at first two gate combinations typical of the QMF state in QCC methods
        num_gates = len(circ._gates)
        if num_gates not in (1, 2):
            circuit_new += circ
            continue

        gate0, gate1 = circ._gates[:2] + [None] * (2 - num_gates)
        gate_0_is_bitflip = is_bitflip_gate(gate0)
        gate_1_is_bitflip = is_bitflip_gate(gate1)

        if num_gates == 1:
            if gate0.name in {"RZ", "Z"}:
                trim_index.append(e_indices[i].pop())
                trim_states.append(0)
            elif gate0.name in {"X", "RX"} and gate_0_is_bitflip:
                trim_index.append(e_indices[i].pop())
                trim_states.append(1)
            else:
                circuit_new += circ
        elif num_gates == 2:
            if gate1.name in {"Z", "RZ"}:
                if gate0.name in {"RZ", "Z"}:
                    trim_index.append(e_indices[i].pop())
                    trim_states.append(0)
                else:
                    circuit_new += circ
            elif gate1.name in {"X", "RX"} and gate_1_is_bitflip:
                if gate0.name in {"RX", "X"} and gate_0_is_bitflip:
                    trim_index.append(e_indices[i].pop())
                    trim_states.append(0)
                elif gate0.name in {"Z", "RZ"}:
                    trim_index.append(e_indices[i].pop())
                    trim_states.append(1)
                else:
                    circuit_new += circ
            else:
                circuit_new += circ
    return circuit_new, trim_index, trim_states

def trim_trivial_qubits(operator, circuit):
    """
        Trim circuit and operator based on expectation values calculated from
        trivial components of the circuit.

        Args:
            operator (QubitOperator): Operator to trim
            circuit (Circuit): circuit to be trimmed
        Returns:
            QubitOperator : Trimmed qubit operator
            Circuit : Trimmed circuit
    """
    trimmed_circuit, trim_index, trim_states = trim_trivial_circuit(circuit)
    trimmed_operator = trim_trivial_operator(operator, trim_index, trim_states, circuit.width, reindex=True)

    return trimmed_operator, trimmed_circuit
