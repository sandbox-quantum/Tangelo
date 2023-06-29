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

"""Module to convert qubit operators to different formats."""

from tangelo.toolboxes.operators import count_qubits, QubitOperator
from tangelo.linq.translator.translate_qiskit import translate_op_from_qiskit, translate_op_to_qiskit
from tangelo.linq.translator.translate_cirq import translate_op_from_cirq, translate_op_to_cirq
from tangelo.linq.translator.translate_qulacs import translate_op_from_qulacs, translate_op_to_qulacs
from tangelo.linq.translator.translate_pennylane import translate_op_from_pennylane, translate_op_to_pennylane
from tangelo.linq.translator.translate_projectq import translate_op_from_projectq, translate_op_to_projectq
from tangelo.linq.translator.translate_sympy import translate_op_to_sympy


FROM_TANGELO = {
    "qiskit": translate_op_to_qiskit,
    "cirq": translate_op_to_cirq,
    "qulacs": translate_op_to_qulacs,
    "pennylane": translate_op_to_pennylane,
    "projectq": translate_op_to_projectq,
    "sympy": translate_op_to_sympy
}

TO_TANGELO = {
    "qiskit": translate_op_from_qiskit,
    "cirq": translate_op_from_cirq,
    "qulacs": translate_op_from_qulacs,
    "pennylane": translate_op_from_pennylane,
    "projectq": translate_op_from_projectq
}


def translate_operator(qubit_operator, source, target, n_qubits=None):
    """Function to convert a qubit operator defined within the "source" format
    to a "target" format.

    Args:
        qubit_operator (source format): Self-explanatory.
        source (string): Identifier for the source format.
        target (string): Identifier for the target format.
        n_qubits (int): Number of qubits relevant to the operator. Not used by
            all format conversion functions, will be computed automatically if
            not specified.

    Returns:
        (operator in target format): Translated qubit operator.
    """

    source = source.lower()
    target = target.lower()

    if source == target:
        return qubit_operator

    if source != "tangelo":
        if source not in TO_TANGELO:
            raise NotImplementedError(f"Qubit operator conversion from {source} to {target} is not supported.")
        qubit_operator = TO_TANGELO[source](qubit_operator)

    if target != "tangelo":
        if target not in FROM_TANGELO:
            raise NotImplementedError(f"Qubit operator conversion from {source} to {target} is not supported.")

        # For translation functions that need an explicit number of qubits.
        if target in {"qiskit", "sympy"}:
            # The count_qubits function has no way to detect the number of
            # qubits when an operator is only a tensor product of I.
            if qubit_operator == QubitOperator((), qubit_operator.constant) and n_qubits is None:
                raise ValueError("The number of qubits (n_qubits) must be provided.")
            n_qubits = count_qubits(qubit_operator) if n_qubits is None else n_qubits
            qubit_operator = FROM_TANGELO[target](qubit_operator, n_qubits)
        else:
            qubit_operator = FROM_TANGELO[target](qubit_operator)

    return qubit_operator
