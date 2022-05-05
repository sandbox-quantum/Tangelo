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

"""Module that defines helper functions to taper qubits in a molecular problem
with Z2 tapering.

For all chemical Hamiltonians, there are at least two symmetries (electron
number and spin conservation) that reduce the qubits count by two. Furthermore,
molecular symmetries can lead to a reduction of the number of qubits required to
encode a problem. Those symmetries can be interpreted as degenerated
eigenvalues. In the real space, symmetry operations lead to the same total
energy.

Ref:
    Tapering off qubits to simulate fermionic Hamiltonians
    Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme.
    arXiv:1701.08213
"""

from functools import reduce
import operator

import numpy as np

from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector
from tangelo.toolboxes.operators.multiformoperator import MultiformOperator, ConvertPauli, do_commute


def get_z2_taper_function(unitary, kernel, q_indices, n_qubits, n_symmetries, eigenvalues=None):
    """Defines a method for applying taper to an arbitrary operator in this
    space. The operator is first conditioned against the tapering-operators:
    terms which do not commute with the Hamiltonian's symmetries are culled, and
    the remaining operator is rotated against the tapering unitary. After
    applying the eigenvalues to account for tapered elements, the operator is
    then finally tapered, with the relevant qubits removed.

    Args:
        unitary (array of float): Unitary matrix to perform U*HU.
        kernel (array of bool): Kernel representing the NULL space for the
            operator.
        q_indices (array of int): Indices for the relevant columns in the
            operator array representation.
        n_qubits (int): Self-explanatory.
        n_symmetries (int): Number of qubits to remove.
        eigenvalues (array of int): Initial state eigenvalues for the qubits
            to be removed.

    Returns:
        function: function for tapering operator.
    """

    def do_taper(operator, eigenvalues=eigenvalues):

        # Remove non-commuting terms.
        commutes = do_commute(operator, kernel, term_resolved=True)
        indices = np.where(commutes is False)[0]

        if len(indices) > 0:
            operator.remove_terms(indices)

        # Apply rotation if the operator is not trivial.
        if operator.n_terms == 0:
            op_matrix, factors = np.zeros(operator.n_qubits), np.array([0.0])
        else:
            product = operator * unitary
            product.compress()
            product_reverse = unitary * product
            product_reverse.compress()
            op_matrix, factors = product_reverse.integer, product_reverse.factors

        if factors.max() == 0.0:
            return MultiformOperator.from_integerop(np.zeros((1, n_qubits-n_symmetries), dtype=int), np.array([0.0]))

        for index, eigenvalue in zip(q_indices, eigenvalues):
            factors[op_matrix[:, index] > 0] *= eigenvalue

        tapered = np.delete(op_matrix, q_indices, axis=1)

        return MultiformOperator.from_integerop(tapered, factors)

    return do_taper


def get_clifford_operators(kernel):
    """Function to identify, with a kernel, suitable single-Pauli gates and the
    related unitary Clifford gates.

    Args:
        kernel (array-like of bool): Array of M x 2N booleans, where M is the
            number of terms and N is the number of qubits. Refers to a qubit
            operator in the stabilizer notation.

    Returns:
        (list of MultiformOperator, list of int): Encoded binary-encoded Pauli
            strings and symmetry indices.
    """

    indices = list()
    n_qubits = kernel.shape[1] // 2
    cliffords = list()
    factors = np.full(2, np.sqrt(0.5))

    for row, ki in enumerate(kernel):
        vector = np.zeros(n_qubits * 2, dtype=int)
        rest = np.delete(kernel, row, axis=0)

        for col in range(n_qubits):
            tau_i = ConvertPauli((ki[col], ki[col+n_qubits])).integer
            tau_j = [ConvertPauli((rj[col], rj[col+n_qubits])).integer for rj in rest]

            # Default value for the conversion (identity). Find a suitable
            # choice of Pauli gate which anti-commutes with one symmetry
            # operator and commutes with all others.
            pauli = (0, 0)
            for pauli_i in range(3):
                tau_destination = np.delete(np.array([1, 2, 3]), pauli_i)
                lookup = [0, pauli_i + 1]

                if all(oi in lookup for oi in tau_j):
                    if tau_i in tau_destination:
                        pauli = ConvertPauli(pauli_i + 1).tuple
                        break

            if sum(pauli) > 0:
                vector[[col, col + n_qubits]] = pauli
                indices.append(col)
                clifford = np.array([vector, ki])
                cliffords.append(MultiformOperator.from_binaryop(bin_op=clifford, factors=factors))
                break

    return cliffords, np.array(indices)


def get_unitary(cliffords):
    """Function generating the product over multiple Clifford
    operators as a single unitary. The result is a MultiformOperator.

    Args:
        cliffords (list of MultiformOperator): Encoded Clifford operators.

    Returns:
        MultiformOperator: Multiplication reflecting the composite operator.
    """
    return reduce(operator.mul, cliffords[1:], cliffords[0])


def get_eigenvalues(symmetries, n_qubits, n_electrons, spin, mapping, up_then_down):
    """Get the initial state eigenvalues, as operated on by each of the symmetry
    operators. These are used to capture the action of each Pauli string in the
    Hamiltonian on the tapered qubits.

    Args:
        symmetries (array-like of bool): Symmetries in binary encoding.
        n_qubits (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        mapping (str): Qubit mapping.
        up_then_down (bool): Whether or not spin ordering is all up then
            all down.

    Returns:
        array of +/-1: Eigenvalues of an operator with symmetries.
    """

    psi_init = get_vector(n_qubits, n_electrons, mapping, up_then_down, spin)

    if len(symmetries.shape) == 1:
        symmetries = np.reshape(symmetries, (-1, len(symmetries)))

    each_qubit = np.einsum("ij,j->ij", symmetries[:, n_qubits:].astype(bool), psi_init)
    eigenvalues = np.product(-2 * each_qubit + 1, axis=1)

    return eigenvalues
