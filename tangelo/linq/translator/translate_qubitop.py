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

"""Module to convert qubit operators to different formats."""

from tangelo.toolboxes.operators import count_qubits
from tangelo.linq.translator.translate_qiskit import quop_qiskit_to_tangelo, quop_tangelo_to_qiskit


FROM_TANGELO = {
    "qiskit": quop_tangelo_to_qiskit
}

TO_TANGELO = {
    "qiskit": quop_qiskit_to_tangelo
}


def translate_operator(qubit_operator, source, target, n_qubits=None):
    """Function to convert a qubit operator defined within the "source" format
    to a "target" format.

    Args:
        qubit_operator (source format): Self-explanatory.
        source (string): Identifier for the source format.
        target (string): Identifier for the target format.
        n_qubits (int): Number of qubits relevant to the operator.

    Returns:
        (target format): Translated qubit operator.
    """

    source = source.lower()
    target = target.lower()

    if source != "tangelo":
        if source not in TO_TANGELO:
            raise NotImplementedError(f"Source {source} is not supported.")
        qubit_operator = TO_TANGELO[source](qubit_operator)
    if target != "tangelo":
        if target not in FROM_TANGELO:
            raise NotImplementedError(f"Target {target} is not supported.")
        n_qubits = count_qubits(qubit_operator) if n_qubits is None else n_qubits
        qubit_operator = FROM_TANGELO[target](qubit_operator, n_qubits)

    return qubit_operator
