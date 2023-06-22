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

"""This module provides functions to create symmetry verification and post-selection circuits."""

import warnings

from tangelo.linq import Circuit, Gate
from tangelo.linq.helpers import measurement_basis_gates, pauli_string_to_of
from tangelo.toolboxes.operators import QubitOperator, count_qubits
from tangelo.toolboxes.post_processing import Histogram


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
            This appends an additional qubit to the circuit.
    """

    if isinstance(sym_op, QubitOperator):
        op_len = count_qubits(sym_op)
    else:
        op_len = len(sym_op)

    n_qubits = circuit.width

    # Check if the operator size matches the number of qubits
    if n_qubits < op_len:
        raise RuntimeError("The size of the symmetry operator is bigger than the number of qubits.")
    elif n_qubits > op_len:
        warnings.warn(
            "The size of the symmetry operator is smaller than the number of qubits. Remaining qubits will be measured in the Z-basis.")

    if isinstance(sym_op, str):
        basis_gates = measurement_basis_gates(pauli_string_to_of(sym_op))
    elif isinstance(sym_op, (list, tuple)):
        basis_gates = measurement_basis_gates(sym_op)
    elif isinstance(sym_op, QubitOperator):
        basis_gates = measurement_basis_gates(list(sym_op.terms.keys())[0])
    else:
        raise RuntimeError(
            "The symmetry operator must be an OpenFermion-style operator, a QubitOperator, or a Pauli word.")

    basis_circ = Circuit(basis_gates)
    parity_gates = [Gate("CNOT", n_qubits, i) for i in range(n_qubits)]
    circuit_new = circuit + basis_circ + Circuit(parity_gates) + basis_circ.inverse()
    return circuit_new


def post_select(freqs, expected_outcomes):
    """Apply post selection to frequency data based on
    a dictionary of expected outcomes on ancilla qubits.

    Args:
        freqs (dict): A dictionary of {bitstring: frequency} pairs
        expected_outcomes (dict): Desired outcomes on certain qubit indices
            and their expected state. For example, {0: "1", 1: "0"} would
            filter results based on the first qubit with the |1> state and
            the second qubit with the |0> state measured.

    Returns:
        dict: A dictionary of post-selected, renormalized frequencies
            and bitstrings with removed ancilla qubits
    """

    hist = Histogram(freqs, n_shots=0)
    hist.post_select(expected_outcomes)
    return hist.frequencies


def strip_post_selection(freqs, *qubits):
    """Convenience function to remove the symmetry ancilla qubit
    and aggregate data to recreate results without post-selection.

    Args:
        freqs (dict): A dictionary of {bitstring: frequency} as returned from a quantum device
        qubits (variable number of int): The ancilla qubit indices to be removed from the bitstrings

    Returns:
        dict: A frequency dictionary with the qubits stripped and data aggregated
    """

    hist = Histogram(freqs, n_shots=0)
    hist.remove_qubit_indices(*qubits)
    return hist.frequencies


def split_frequency_dict(frequencies, indices, desired_measurement=None):
    """Marginalize the frequencies dictionary over the indices.
    This splits the frequency dictionary into two frequency dictionaries
    and aggregates the corresponding frequencies.
    If desired_measurement is provided, the marginalized frequencies are
    post-selected for that outcome on the mid-circuit measurements.

    Args:
        frequencies (dict): The input frequency dictionary
        indices (list): The list of indices in the frequency dictionary to marginalize over
        desired_measurement (str): The bit string that is to be selected

    Returns:
        dict: The marginal frequencies for provided indices
        dict: The marginal frequencies for remaining indices
    """

    key_length = len(next(iter(frequencies)))
    other_indices = [i for i in range(key_length) if i not in indices]

    midcirc_dict = strip_post_selection(frequencies, *other_indices)

    if desired_measurement is None:
        marginal_dict = strip_post_selection(frequencies, *indices)
    else:
        expected_outcomes = {i: m for i, m in zip(indices, desired_measurement)}
        marginal_dict = post_select(frequencies, expected_outcomes)

    return midcirc_dict, marginal_dict
