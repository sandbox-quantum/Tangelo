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

"""Modules to taper qubits in a molecular problem. This is defined in the
litterature as Z2 taperring. For all problem, there are at least two symmetries
(electron number and spin conservation) that reduce the qubits count by two.
Furthermore, molecular symmetries can lead to a reduction of the qubit
required for a problem. Thoses symmetries can be interpreted as degenerated
eigenvalues. In the real space, symmetry operations lead to the same total
energy. For example, the water molecule (C2v group) energy is invariant to a
180 deg rotation around the z axis.
Here are some important references:
- Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz
  Jonathan Romero, Ryan Babbush, Jarrod R. McClean, Cornelius Hempel, Peter Love, Alán Aspuru-Guzik
  arXiv:1701.02691
  https://arxiv.org/pdf/1701.02691.pdf
- Tapering off qubits to simulate fermionic Hamiltonians
  Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme
  arXiv:1701.08213
  https://arxiv.org/pdf/1701.08213.pdf
"""

#import pyscf # Resolve the "AttributeError: module 'importlib' has no attribute 'abc'"" !?
from itertools import count
import numpy as np
from operator import itemgetter

from tangelo.toolboxes.operators import QubitOperator, HybridOperator
from tangelo.toolboxes.operators.operators import count_qubits
from tangelo.toolboxes.operators.hybridoperator import ConvertPauli


def get_clifford_operators(kernel):
    """Utilize the kernel of the operator to identify suitable single-pauli gates
    and the related unitary Clifford gates.
    Args:
        kernel (array-like of bool): Numpy array of Mx2N (M: number of terms,
            N: number of qubits) binary int, symmetries.
    Returns:
        list of HybridOperator: Encoded binary-encoded Pauli strings.
        list of int: Symmetry indices.
    """

    indices = []
    n_qubits = kernel.shape[1] // 2
    cliffords = []
    factors = np.array([np.sqrt(0.5), np.sqrt(0.5)])

    for row, ki in enumerate(kernel):
        vector = np.zeros(n_qubits*2,dtype=int)
        rest = np.delete(kernel,row,axis=0)
        for col in range(n_qubits):
            tau_i = ConvertPauli((ki[col], ki[col+n_qubits])).integer
            tau_j = [ConvertPauli((rj[col], rj[col+n_qubits])).integer for rj in rest]

            # Default value for the conversion (identity).
            # Find a suitable choice of Pauli-gate which anti-commutes with one
            # symmetry operator and commutes with all others.
            pauli = (0,0)
            for pauli_i in range(3):
                tau_destination = np.delete(np.array([1,2,3]), pauli_i)
                lookup = [0, pauli_i +1]

                if all(oi in lookup for oi in tau_j):
                    if tau_i in tau_destination:
                        pauli = ConvertPauli(pauli_i+1).tuple
                        break

            if sum(pauli) > 0:
                vector[[col,col+n_qubits]] = pauli
                indices.append(col)
                clifford_int = np.array([vector,ki])
                cliffords.append(HybridOperator(n_qubits, binary_operator=clifford_int, factors=factors))
                break

    return cliffords, np.array(indices)

def get_unitary(cliffords):
    """Recursive function for generating the product over multiple Clifford operators
    as a single unitary. The result is a HybridOperator.
    Args:
        cliffords (list of HybridOperator):  Encoded cliffors operators.
    Returns:
        HybridOperator: Multiplication reflecting the composite operator.
    """

    # Recursion function.
    if len(cliffords)>2:
        return cliffords[0] * get_unitary(cliffords[1:])
    elif len(cliffords) == 2:
        return cliffords[0] * cliffords[1]
    elif len(cliffords) == 1:
        return cliffords[0]

def get_eigenvalues(symmetries, n_qubits, n_electrons, mapping='JW'):
    """Get the initial state eigenvalues, as operated on by each of the symmetry
    operators. These are used to capture the action of each Pauli string in the
    Hamiltonian on the tapered qubits.
    TODO: (Ryan) Examine plausibility of utilizing z-portion of the symmetry
        operators.
    Args
        symmetries (array-like of bool): Symmetries in binary encoding.
        n_qubits (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        mapping (str): Mapping process used for the fermion->qubit.
    Returns:
        array of +/-1: Eigenvalues of operator with symmetries.
    """

    vector = np.zeros(n_qubits, dtype=bool)
    vector[:n_electrons] = True
    #if mapping == 'BK':
    #    vector = do_bk_transform(vector).astype(bool)

    psi_init = vector

    if len(symmetries.shape) == 1:
        symmetries = np.reshape(symmetries, (-1, len(symmetries)))

    each_qubit = np.einsum('ij,j->ij', symmetries[:,n_qubits:].astype(bool), psi_init)
    eigenvalues = np.product(-2*each_qubit+1, axis=1)

    return eigenvalues

def collapse(operator, factors):
    """Identify and sum over duplicate terms in an operator, to collapse
    a set of PauliStrings to their minimal representation.
    Returns the unique array of integer-encoded PauliStrings, their factors
    in the operator, and their indices
    """
    all_terms = np.concatenate((operator, np.linspace(0, len(operator)-1, len(operator), dtype=int).reshape(len(operator), -1)), axis=1)
    qubits = np.linspace(0, np.shape(operator)[1]-1, np.shape(operator)[1], dtype=int)
    sorted_terms = np.array(sorted(all_terms, key=itemgetter(*qubits)))
    sorted_factors = factors[sorted_terms[:,-1]]
    unique, _, inverse = np.unique(sorted_terms[:,:-1], axis=0, return_inverse=True, return_index=True)
    unique = unique.astype(int)
    factors = np.zeros(len(unique), dtype=complex)
    for index in range(len(sorted_terms)):
        factors[inverse[index]] += sorted_factors[index]

    nonzero = np.where(abs(factors) > 0)
    unique = unique[nonzero]
    factors = factors[nonzero]
    if len(np.shape(unique)) == 1:
        unique = np.reshape(unique, (-1, len(unique)))

    return unique, factors

def get_taper(unitary, kernel, q_indices, n_qubits, n_symmetry, eigenvalues=None):
    """
    Define method for applying taper to an arbitrary operator in this space.
    The operator is first conditioned against the tapering-operators: terms
    which do not commute with the Hamiltonian's symmetries are culled, and the
    remaining operator is rotated against the tapering unitary. After applying
    the eigenvalues to account for tapered elements, the operator is then finally
    tapered, with the relevant qubits removed.
    *return*:
            - **do_taper**: method for tapering operator
    """
    def do_taper(operator, eigenvalues=eigenvalues):
        """
        Tapering method.
        *args*:
                - **operator**: HybridOperator to taper
                - **eigenvalues**: numpy array of float
        *return*:
                - HybridOperator with qubit number reduced.
        """

        # Remove non-commuting terms.
        # Minimize # of iterations, use shorter operator.
        commutes = operator.commutes_with(kernel, term_resolved=True)
        indices = np.where(commutes == False)[0]
        if len(indices) > 0:
            operator.remove_terms(indices)
        if operator.n_terms == 0:
            operator_matrix, factors = np.zeros(operator.n_qubits), np.array([0.0])
        else:
            #apply rotation
            product = operator * unitary

            product_reverse = unitary * product
            post, factors =  product_reverse.integer, product_reverse.factors
            #clean operator
            operator_matrix, factors = collapse(post, factors)

        if factors.max() == 0.0:
            return HybridOperator(n_qubits-n_symmetry, integer_operator=np.zeros((1, n_qubits-n_symmetry), dtype=int), factors=np.array([0.0]))

        for ii, index in enumerate(q_indices):
            factors[operator_matrix[:,index]>0] *= eigenvalues[ii]

        tapered = np.delete(operator_matrix, q_indices, axis=1)
        return HybridOperator(n_qubits-n_symmetry, integer_operator=tapered, factors=factors)

    return do_taper

def taper_hamiltonian(qubit_hamiltonian, n_qubits, n_electrons):
    op1 = HybridOperator(n_qubits, qubit_operator=qubit_hamiltonian)

    kernel = op1.get_kernel()
    list_of_cliffords, q_indices = get_clifford_operators(kernel)
    unitary = get_unitary(list_of_cliffords)
    kernel_operator = HybridOperator(n_qubits, binary_operator=kernel, factors=np.ones(len(kernel)))

    n_symmetry = len(q_indices)
    eigenvalues = get_eigenvalues(kernel_operator.binary, n_qubits, n_electrons, "JW")

    taper = get_taper(unitary, kernel_operator, q_indices, n_qubits, n_symmetry, eigenvalues)

    tap_H = taper(op1)

    return tap_H.integer_to_qubit(), taper
