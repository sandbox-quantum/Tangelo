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

"""Helper function: pauli word rotations."""

import numpy as np
from tangelo.linq import Gate


def measurement_basis_gates(term):
    """Generate the rotation gates to perform change of basis before
    measurement.

    Args:
        term: Openfermion-style term. Essentially a list of (int, str) tuples.

    Returns:
        list of Gate: A list containing the rotation gates.
    """
    gates = []
    for qubit_index, pauli in term:
        if pauli in {"I", "Z"}:
            pass
        elif pauli == "X":
            gates.append(Gate("RY", qubit_index, parameter=-np.pi/2))
        elif pauli == "Y":
            gates.append(Gate("RX", qubit_index, parameter=np.pi/2))
        else:
            raise RuntimeError("Measurement basis not supported (currently supporting I,X,Y,Z)")
    return gates


def pauli_string_to_of(pauli_string):
    """ Converts a string of I,X,Y,Z Pauli operators to an Openfermion-style
    representation.
    """
    return [(i, p) for i, p in enumerate(pauli_string) if p != 'I']


def pauli_of_to_string(pauli_op, n_qubits):
    """ Converts an Openfermion-style Pauli word to a string representation.
    The user must specify the total number of qubits.
    """
    p_string = ['I'] * n_qubits
    for i, p in pauli_op:
        p_string[i] = p
    return ''.join(p_string)
