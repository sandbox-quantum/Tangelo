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

"""Modules that defines helper functions to taper qubits in a molecular problem
with Z2 tapering.

For all problem, there are at least two symmetries (electron number
and spin conservation) that reduce the qubits count by two. Furthermore,
molecular symmetries can lead to a reduction of the qubit required for a problem.
Thoses symmetries can be interpreted as degenerated eigenvalues. In the real
space, symmetry operations lead to the same total energy.

Ref:
Tapering off qubits to simulate fermionic Hamiltonians
Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme.
arXiv:1701.08213
"""

from operator import itemgetter

import numpy as np

from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector
from tangelo.toolboxes.operators.hybridoperator import HybridOperator, ConvertPauli, is_commuting


def get_z2_taper_function(unitary, kernel, q_indices, n_qubits, n_symmetries, eigenvalues=None):
    """Defines a method for applying taper to an arbitrary operator in this
    space. The operator is first conditioned against the tapering-operators:
    terms which do not commute with the Hamiltonian's symmetries are culled, and
    the remaining operator is rotated against the tapering unitary. After
    applying the eigenvalues to account for tapered elements, the operator is
    then finally tapered, with the relevant qubits removed.

    Returns:
        function: function for tapering operator.
    """

    def do_taper(operator, eigenvalues=eigenvalues):

        # Remove non-commuting terms.
        commutes = is_commuting(operator, kernel, term_resolved=True)
        indices = np.where(commutes == False)[0]

        if len(indices) > 0:
            operator.remove_terms(indices)

        # Apply rotation if the operator is not trivial.
        if operator.n_terms == 0:
            operator_matrix, factors = np.zeros(operator.n_qubits), np.array([0.0])
        else:
            product = operator * unitary
            product_reverse = unitary * product
            post, factors =  product_reverse.integer, product_reverse.factors

            #Clean operator.
            operator_matrix, factors = collapse(post, factors)

        if factors.max() == 0.0:
            return HybridOperator.from_integerop(np.zeros((1, n_qubits-n_symmetries), dtype=int), np.array([0.0]))

        for index, eigenvalue in zip(q_indices, eigenvalues):
            factors[operator_matrix[:, index] > 0] *= eigenvalue

        tapered = np.delete(operator_matrix, q_indices, axis=1)

        return HybridOperator.from_integerop(tapered, factors)

    return do_taper


def get_clifford_operators(kernel):
    """Function to identify, with a kernel, suitable single-pauli gates and the
    related unitary Clifford gates.

    Args:
        kernel (array-like of bool): Array of M x 2N booleans, where M is the
            number of terms and N is the number of qubits. Refers to a qubit
            operator in the stabilizer notation.

    Returns:
        (list of HybridOperator, list of int): Encoded binary-encoded Pauli
            strings and symmetry indices.
    """

    indices = list()
    n_qubits = kernel.shape[1] // 2
    cliffords = list()
    factors = np.array([np.sqrt(0.5), np.sqrt(0.5)])

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
                tau_destination = np.delete(np.array([1,2,3]), pauli_i)
                lookup = [0, pauli_i +1]

                if all(oi in lookup for oi in tau_j):
                    if tau_i in tau_destination:
                        pauli = ConvertPauli(pauli_i + 1).tuple
                        break

            if sum(pauli) > 0:
                vector[[col, col + n_qubits]] = pauli
                indices.append(col)
                clifford = np.array([vector, ki])
                cliffords.append(HybridOperator.from_binaryop(bin_op=clifford, factors=factors))
                break

    return cliffords, np.array(indices)


def get_unitary(cliffords):
    """Recursive function for generating the product over multiple Clifford
    operators as a single unitary. The result is a HybridOperator.

    Args:
        cliffords (list of HybridOperator): Encoded cliffors operators.

    Returns:
        HybridOperator: Multiplication reflecting the composite operator.
    """

    # Recursion algorithm.
    if len(cliffords) > 2:
        return cliffords[0] * get_unitary(cliffords[1:])
    elif len(cliffords) == 2:
        return cliffords[0] * cliffords[1]
    else:
        return cliffords[0]


def get_eigenvalues(symmetries, n_qubits, n_electrons, mapping, up_then_down):
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
        array of +/-1: Eigenvalues of operator with symmetries.
    """

    psi_init = get_vector(n_qubits, n_electrons, mapping, up_then_down)

    if len(symmetries.shape) == 1:
        symmetries = np.reshape(symmetries, (-1, len(symmetries)))

    each_qubit = np.einsum("ij,j->ij", symmetries[:,n_qubits:].astype(bool), psi_init)
    eigenvalues = np.product(-2 * each_qubit + 1, axis=1)

    return eigenvalues


def collapse(operator, factors):
    """Function to identify and sum over duplicate terms in an operator, to
    collapse a set of Pauli words to their minimal representation.

    Returns:
    (array of int, arrays of float):A rray of unique integer-encoded
        Pauli words, their factors in the operator.
    """

    all_terms = np.concatenate((operator, np.linspace(0, len(operator) - 1, len(operator), dtype=int).reshape(len(operator), -1)), axis=1)

    qubits = np.linspace(0, operator.shape[1] - 1, operator.shape[1], dtype=int)

    sorted_terms = np.array(sorted(all_terms, key=itemgetter(*qubits)))
    sorted_factors = factors[sorted_terms[:,-1]]

    unique, inverse = np.unique(sorted_terms[:,:-1], axis=0, return_inverse=True)

    factors = np.zeros(len(unique), dtype=complex)

    for index in range(len(sorted_terms)):
        factors[inverse[index]] += sorted_factors[index]

    nonzero = np.where(abs(factors) > 0)
    unique = unique[nonzero]
    factors = factors[nonzero]

    if len(np.shape(unique)) == 1:
        unique = np.reshape(unique, (-1, len(unique)))

    return unique.astype(int), factors
