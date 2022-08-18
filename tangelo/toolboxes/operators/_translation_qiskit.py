# Copyright 2021 Good Chemistry Company.
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

"""Module to convert Qiskit qubit operators."""

from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq.helpers import pauli_of_to_string, pauli_string_to_of


def tangelo_to_qiskit(qubit_operator, n_qubits):
    """Helper function to translate a Tangelo QubitOperator to a qiskit
    PauliSumOp. Qiskit must be installed for the function to work.

    Args:
        qubit_operator (tangelo.toolboxes.operators.QubitOperator): Self-explanatory.
        n_qubits (int): Number of qubits relevant to the operator.

    Returns:
        (qiskit.opflow.primitive_ops.PauliSumOp): Qiskit qubit operator.
    """

    # Import qiskit functions.
    from qiskit.opflow.primitive_ops import PauliSumOp

    # Convert each term sequencially.
    term_list = list()
    for term_tuple, coeff in qubit_operator.terms.items():
        term_string = pauli_of_to_string(term_tuple, n_qubits)

        # Reverse the string becasue of qiskit convention.
        term_list += [(term_string[::-1], coeff)]

    return PauliSumOp.from_list(term_list)


def qiskit_to_tangelo(qubit_operator):
    """Helper function to translate a a qiskit PauliSumOp to a Tangelo
    QubitOperator.

    Args:
        qubit_operator (qiskit.opflow.primitive_ops.PauliSumOp): Self-explanatory.

    Returns:
        (tangelo.toolboxes.operators.QubitOperator): Tangelo qubit operator.
    """

    # Creation of a dictionary to append all terms at once.
    terms_dict = dict()
    for pauli_word in qubit_operator:
        # Inversion of the string because of qiskit ordering.
        term_string = pauli_word.to_pauli_op().primitive.to_label()[::-1]
        term_tuple = pauli_string_to_of(term_string)
        terms_dict[tuple(term_tuple)] = pauli_word.coeff

    # Create and copy the information into a new QubitOperator.
    tangelo_op = QubitOperator()
    tangelo_op.terms = terms_dict

    # Clean the QubitOperator.
    tangelo_op.compress()

    return tangelo_op
