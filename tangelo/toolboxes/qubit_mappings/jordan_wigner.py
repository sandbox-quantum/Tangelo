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

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Jordan-Wigner transform on fermionic operators."""

from openfermion.transforms import jordan_wigner as openfermion_jordan_wigner


def jordan_wigner(operator):
    r"""Apply the Jordan-Wigner transform to a FermionOperator,
    InteractionOperator, or DiagonalCoulombHamiltonian to convert to a
    QubitOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Returns:
        QubitOperator: An instance of the QubitOperator class.

    Warning:
        The runtime of this method is exponential in the maximum locality of the
        original FermionOperator.

    Raises:
        TypeError: Operator must be a FermionOperator,
        DiagonalCoulombHamiltonian, or InteractionOperator.
    """
    qubit_operator = openfermion_jordan_wigner(operator)

    return qubit_operator
