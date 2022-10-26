# Copyright 2022 Good Chemistry Company.
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

"""This module provides functions to create symmetry verification and post-selection circuits."""

import warnings

from tangelo.linq import Circuit, Gate
from tangelo.linq.helpers import measurement_basis_gates, pauli_string_to_of
from tangelo.toolboxes.operators import QubitOperator, count_qubits


def ancilla_symmetry_circuit(circuit, sym_op):
    """Append a symmetry operator circuit to an input circuit using an extra ancilla qubit,
    for the purpose of symmetry verification and post-selection.
    For more details see arXiv:1807.10050.

    Args:
        circuit (Circuit): A quantum circuit to be equipped with symmetry verification
        sym_op (string): The symmetry operator to be verified.
            Can be a Pauli string, OpenFermion style list or a QubitOperator

    Returns:
        Circuit: The input circuit appended with the proper basis rotation
            and entanglement with an ancilla qubit, which is added as the last qubit.
            Increases the input circuit width by 1."""
    if isinstance(sym_op, QubitOperator):
        op_len = count_qubits(sym_op)
    else:
        op_len = len(sym_op)
    n_qubits = circuit.width

    # Check if the operator size matches the circuit width
    if n_qubits < op_len:
        raise RuntimeError("The size of the symmetry operator is bigger than the circuit width.")
    elif n_qubits > op_len:
        warnings.warn("The size of the symmetry operator is smaller than the circuit width. Remaining qubits will be measured in the Z-basis.")

    if isinstance(sym_op, str):
        basis_gates = measurement_basis_gates(pauli_string_to_of(sym_op))
    elif isinstance(sym_op, list or tuple):
        basis_gates = measurement_basis_gates(sym_op)
    elif isinstance(sym_op, QubitOperator):
        basis_gates = measurement_basis_gates(list(sym_op.terms.keys())[0])

    basis_circ = Circuit(basis_gates)
    parity_gates = [Gate("CNOT", n_qubits, i) for i in range(n_qubits)]
    circuit_new = circuit + basis_circ + Circuit(parity_gates) + basis_circ.inverse()
    return circuit_new


def post_select(hist, expected_outcomes):
    """Apply post selection to frequency data based on
    a dictionary of expected outcomes on ancilla qubits.

    Args:
        hist (dict): A dictionary of {bitstring: frequency} pairs
        expected_outcomes (dict): Desired outcomes on certain qubit indices
            and their expected state. For example, {0: "1", 1: "0"} would
            filter results based on the first qubit with the |1> state and
            the second qubit with the |0> state measured.

    Returns:
        dict: A dictionary of post-selected, renormalized frequencies
            and bitstrings with removed ancilla qubits
    """
    def f_post_select(bitstring):
        for qubit_i, expected_bit in expected_outcomes.items():
            if bitstring[qubit_i] != expected_bit:
                return False
        return True

    counts_new = {}

    for string, freq in hist.items():
        if f_post_select(string):
            new_str = "".join([s for i, s in enumerate(string) if i not in expected_outcomes.keys()])
            counts_new[new_str] = freq

    factor = sum(counts_new.values())
    counts_new = {key: value/factor for key, value in counts_new.items()}
    return counts_new


def strip_post_selection(freqs, qubits):
    """Convenience function to remove the symmetry ancilla qubit
    and aggregate data to recreate results without post-selection.

    Args:
        freqs (dict): A dictionary of {bitstring: frequency} as returned from a quantum device
        qubits (int or list): An ancilla qubit index, or list of indices, to be removed from the bitstrings

    Returns:
        dict: A frequency dictionary with the qubits stripped and data aggregated"""
    if isinstance(qubits, int):
        qubits = [qubits]

    counts_new = {}

    for string, counts in freqs.items():
        new_str = "".join([s for i, s in enumerate(string) if i not in qubits])
        counts_new[new_str] = counts_new.get(new_str, 0) + counts

    return counts_new
