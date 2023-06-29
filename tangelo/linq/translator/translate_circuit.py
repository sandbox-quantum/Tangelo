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

"""Module to convert quantum circuits to different formats."""

from tangelo.linq.translator.translate_braket import translate_c_to_braket, translate_c_from_braket
from tangelo.linq.translator.translate_cirq import translate_c_to_cirq
from tangelo.linq.translator.translate_json_ionq import translate_c_to_json_ionq, translate_c_from_json_ionq
from tangelo.linq.translator.translate_openqasm import translate_c_to_openqasm, translate_c_from_openqasm
from tangelo.linq.translator.translate_projectq import translate_c_to_projectq, translate_c_from_projectq
from tangelo.linq.translator.translate_qdk import translate_c_to_qsharp
from tangelo.linq.translator.translate_qiskit import translate_c_to_qiskit, translate_c_from_qiskit
from tangelo.linq.translator.translate_qulacs import translate_c_to_qulacs
from tangelo.linq.translator.translate_pennylane import translate_c_to_pennylane
from tangelo.linq.translator.translate_sympy import translate_c_to_sympy
from tangelo.linq.translator.translate_stim import translate_c_to_stim

FROM_TANGELO = {
    "braket": translate_c_to_braket,
    "cirq": translate_c_to_cirq,
    "ionq": translate_c_to_json_ionq,
    "openqasm": translate_c_to_openqasm,
    "projectq": translate_c_to_projectq,
    "qdk": translate_c_to_qsharp,
    "qiskit": translate_c_to_qiskit,
    "qulacs": translate_c_to_qulacs,
    "pennylane": translate_c_to_pennylane,
    "stim": translate_c_to_stim,
    "sympy": translate_c_to_sympy
}

TO_TANGELO = {
    "braket": translate_c_from_braket,
    "ionq": translate_c_from_json_ionq,
    "openqasm": translate_c_from_openqasm,
    "projectq": translate_c_from_projectq,
    "qiskit": translate_c_from_qiskit
}


def translate_circuit(circuit, target, source="tangelo", output_options=None):
    """Function to convert a quantum circuit defined within the "source" format
    to a "target" format.

    Args:
        circuit (source format): Self-explanatory.
        target (string): Identifier for the target format.
        source (string): Identifier for the source format.
        output_options (dict): Backend specific options (e.g. a noise model,
            number of qubits, etc.).

    Returns:
        (circuit in target format): Translated quantum circuit.
    """

    source = source.lower()
    target = target.lower()

    if output_options is None:
        output_options = dict()

    if source == target:
        return circuit

    # Convert to Tangelo format if necessary.
    if source != "tangelo":
        if source not in TO_TANGELO:
            raise NotImplementedError(f"Circuit conversion from {source} to {target} is not supported.")
        circuit = TO_TANGELO[source](circuit)

    # Convert to another target format if necessary.
    if target != "tangelo":
        if target not in FROM_TANGELO:
            raise NotImplementedError(f"Circuit conversion from {source} to {target} is not supported.")
        circuit = FROM_TANGELO[target](circuit, **output_options)

    return circuit
