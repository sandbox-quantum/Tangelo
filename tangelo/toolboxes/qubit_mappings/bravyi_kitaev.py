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

"""Tools for performing Bravyi-Kitaev Transformation, as prescribed via the
Fenwick Tree mapping. This implementation accommodates mapping of qubit
registers where the number of qubits is not a power of two.

NB: this is a minimal implementation, just wrapping around current openfermion
code (v 1.0.1). This wrapper enables future refactoring to utilize our own
implementation, as needed.
"""

from openfermion.utils import count_qubits
from openfermion.transforms import bravyi_kitaev as openfermion_bravyi_kitaev


def bravyi_kitaev(fermion_operator, n_qubits):
    """Execute transformation of FermionOperator to QubitOperator using the
    Bravyi-Kitaev transformation. Important note: there are several
    implementations of "Bravyi Kitaev" transformation, in both the literature,
    and historical versions of openfermion. This function executes the
    transformaton defined in arXiv:quant-ph/0003137. Different versions are not
    necessarily the same, and result in undesirable performance. This method is
    a simple wrapper around openfermion's bravyi_kitaev, but we are forcing the
    user to pass n_qubits to avoid unexpected behaviour.

    Args:
        fermion_operator (FermionOperator): input fermionic operator to be
            transformed.
        n_qubits (int): number of qubits associated with the operator.

    Returns:
        QubitOperator: output bravyi-kitaev encoded qubit operator.
    """
    if not (type(n_qubits) is int):
        raise TypeError("Number of qubits (n_qubits) must be integer type.")
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError("Invalid (too few) number of qubits (n_qubits) for input operator.")

    qubit_operator = openfermion_bravyi_kitaev(fermion_operator, n_qubits=n_qubits)

    return qubit_operator
