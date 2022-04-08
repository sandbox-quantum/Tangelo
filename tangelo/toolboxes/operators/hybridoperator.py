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

"""Docdocdoc.
"""

import numpy as np

from tangelo.helpers.math import bool_col_echelon
from tangelo.toolboxes.operators import QubitOperator, count_qubits


class HybridOperator(QubitOperator):
    """Construct integer and binary representations of the operator, as based on
    the user-specified input. Internal operations are mostly done with an
    integer array. The conversion is defined as:
      Letter | Integer | Binary
    -    I   |    0    | (0,0)
    -    Z   |    1    | (0,1)
    -    X   |    2    | (1,0)
    -    Y   |    3    | (1,1)
    Most of the algorithm use the integer representation to detect symmetries
    and performed operation (ex: multiplication). At the end, the operator can
    be output as a QubitOperator.

    Attributes:
        terms (dict): Qubit terms.
        n_qubits (int): Number of qubits for this operator.
        factors (array-like): Factors fopr each term. Without that, information
            would be lost when converting from a QubitOperator.
        integer (array-like): Array of 0s, 1s, 2s and 3s. Each line represents a
            term, and each term is a Pauli word.
        binary (array-like): Array of 0s (false) and 1s (true). Binary
            representation of the integer array.
        binary_swap (array-like): Copy of self.binary but with swapped columns.
        kernel (array-like): Null space of the binary representation + identity.

    Properties:
        n_terms (int): Number of terms in this qubit Hamiltonian.
    """

    def __init__(self, terms, n_qubits, factors, integer, binary):

        # Parent class definition.
        super(QubitOperator, self).__init__()
        self.terms = terms
        self.n_qubits = n_qubits

        self.factors = factors
        self.integer = integer
        self.binary = binary

        # Swapping the binary representation. Example: instead of (0,1), write
        # (1,0). This is useful to detect commutation between terms.
        list_col_swap = [n_column for n_column in range(2 * n_qubits)]
        list_col_swap = list_col_swap[n_qubits:] + list_col_swap[:n_qubits]
        self.binary_swap = self.binary[:, list_col_swap]

        # Attribute to be defined when symmetries will be computed.
        self.kernel = None

    @property
    def n_terms(self):
        return len(self.terms)

    @property
    def qubitoperator(self):
        qubit_op = QubitOperator()
        qubit_op.terms = self.terms
        return qubit_op

    @classmethod
    def from_qubitop(cls, qubit_op, n_qubits=None):
        "Initialize HybridOperator from a qubit operator."

        n_qubits = n_qubits if n_qubits is not None else count_qubits(qubit_op)
        factors = np.array([coeff for coeff in qubit_op.terms.values()])
        int_op = qubit_to_integer(qubit_op, n_qubits)
        bin_op = integer_to_binary(int_op)

        return cls(qubit_op.terms, n_qubits, factors, int_op, bin_op)

    @classmethod
    def from_integerop(cls, int_op, factors):
        "Initialize HybridOperator from an integer operator."

        assert len(factors) == int_op.shape[0]
        terms = integer_to_qubit_terms(int_op, factors)
        bin_op = integer_to_binary(int_op)

        return cls(terms, int_op.shape[1], factors, int_op, bin_op)

    @classmethod
    def from_binaryop(cls, bin_op, factors):
        "Initialize HybridOperator from a binary operator."

        assert len(factors) == bin_op.shape[0]
        n_qubits = bin_op.shape[1] // 2
        # The integer array is defined trivially with the binary array.
        int_op = 2 * bin_op[:, :n_qubits] + bin_op[:, n_qubits:]
        terms = integer_to_qubit_terms(int_op, factors)

        return cls(terms, n_qubits, factors, int_op, bin_op)

    def __mul__(self, other_operator):
        """Multiply two HybridOperators together, return a HybridOperator
        corresponding to the product.

        Args:
            other_operator (HybridOperator): Other operator to multiply self.

        Returns:
            HybridOperator: Product of the multiplication.
        """

        # (Guess from Alexandre): Matrix to take into account the order of pauli multiplication.
        levi = np.zeros((4, 4, 4), dtype=complex)
        levi[0, 0, 0] = 1
        levi[0, 1, 1] = 1
        levi[1, 0, 1] = 1
        levi[0, 2, 2] = 1
        levi[2, 0, 2] = 1
        levi[3, 0, 3] = 1
        levi[0, 3, 3] = 1
        levi[1, 1, 0] = 1
        levi[2, 2, 0] = 1
        levi[3, 3, 0] = 1
        levi[1, 2, 3] = 1j
        levi[1, 3, 2] = -1j
        levi[2, 1, 3] = -1j
        levi[2, 3, 1] = 1j
        levi[3, 2, 1] = -1j
        levi[3, 1, 2] = 1j

        factors = np.zeros((self.n_terms * other_operator.n_terms), dtype=complex)
        product = np.zeros((factors.shape[0], self.n_qubits), dtype=int)
        increment = other_operator.n_terms

        for term_i in range(self.n_terms):
            levi_term = levi[self.integer[term_i], other_operator.integer]
            non_zero = np.where(levi_term != 0)

            product[term_i * increment:(term_i + 1) * increment] = non_zero[2].reshape((increment, self.n_qubits))
            factors[term_i * increment:(term_i + 1) * increment] = self.factors[term_i] * \
                                                                   other_operator.factors * \
                                                                   np.product(levi_term[non_zero].reshape((increment, self.n_qubits)), axis=1)
        print(factors)
        return HybridOperator.from_integerop(product, factors=factors)

    def get_kernel(self):
        """Get the kernel for a matrix of binary integer. Identity matrix is
        appended, and the extended matrix is then reduced to column-echelon
        form. The null space is then extracted.

        Returns:
            Array-like: 2d numpy array of binary integer.
        """

        identity_to_append = np.identity(2*self.n_qubits)

        # Append identity to the matrix E.
        E_prime = np.concatenate((self.binary, identity_to_append), axis=0)

        # Put the matrix into column-echelon form.
        E_prime = E_prime.astype(bool)
        E_prime = bool_col_echelon(E_prime)

        # Extracting the kernel.
        kernel = np.array([E_prime[self.binary.shape[0]:, i] for i in range(self.n_qubits)
                           if not E_prime[:self.binary.shape[0], i].max()])
        kernel = np.concatenate((kernel[:, self.n_qubits:], kernel[:,: self.n_qubits]), axis=1)

        self.kernel = kernel
        return kernel

    def remove_terms(self, indices):
        """Remove term(s) from operator. The associated entry is deleted from
        factors, integer, binary and binary_swap.

        Args:
            indices (int or list of int): Remove all terms corresponding to the
                indices.
        """

        if isinstance(indices, int):
            indices = np.array([indices])

        self.factors = np.delete(self.factors, indices, axis=0)
        self.integer = np.delete(self.integer, indices, axis=0)
        self.binary = np.delete(self.binary, indices, axis=0)
        self.binary_swap = np.delete(self.binary_swap, indices, axis=0)

        # Update the terms attribute.
        self.terms = integer_to_qubit_terms(self.integer, self.factors)


class ConvertPauli:
    """Helper class to help converting from/to string <-> int <-> stabilizer. It
    aims to replace multiple dictionaries. It performs the conversion of
    I <-> 0 <-> (0, 0)
    Z <-> 1 <-> (0, 1)
    X <-> 2 <-> (1, 0)
    Y <-> 3 <-> (1, 1)
    The input can be whatever found in the previous conversion table.

    Args:
        pauli_id (string, int or tuple of bool): Single Pauli.

    Attributes:
        char (string): Pauli letter.
        integer (int): Pauli integer.
        tuple (tuple of bool): Pauli binary representation.
    """

    def __init__(self, pauli_id):
        pauli_translation = [["I", 0, (0, 0)],
                             ["Z", 1, (0, 1)],
                             ["X", 2, (1, 0)],
                             ["Y", 3, (1, 1)]
                            ]

        self.char = None
        self.integer = None
        self.tuple = None

        # Going through the equivalences. When an input is found, attributes are
        # defined.
        for equiv in pauli_translation:
            if pauli_id in equiv:
                self.char = equiv[0]
                self.integer = equiv[1]
                self.tuple = equiv[2]

        # When the input is wrong (ex: "x", "G", (3,4), 42, ...), it raises an error.
        if self.char is None:
            raise ValueError(f"{pauli_id} is an unknown Pauli id.")


def qubit_to_integer(qubit_op, n_qubits=None):
    """Perform conversion from qubit operator to integer array. The integer
    attribute is instantiated, and populated from the QubitOperator attribute.

    Args:
        array-like of int: 2-D numpy array of 0s (I), 1s (Z), 2s (X) and 3s (Y)
            representing a qubit operator (shape[0] = number of terms, shape[1]
            = number of qubits.
    """

    n_qubits = n_qubits if n_qubits is not None else count_qubits(qubit_op)
    integer = np.zeros((len(qubit_op.terms), n_qubits), dtype=int)

    for index, term in enumerate(qubit_op.terms):
        for n_qubit, pauli_letter in term:
            integer[index, n_qubit] = ConvertPauli(pauli_letter).integer

    return integer


def integer_to_binary(integer_op):
    """Perform conversion from integer array to binary (stabilizer) array.

    Returns:
        array-like of bool: Array of 0s and 1s representing the operator with
            the stabilizer notation.
    """

    n_terms = integer_op.shape[0]
    n_qubits = integer_op.shape[1]

    binary = np.zeros((n_terms, 2*n_qubits), dtype=bool)

    binary_x = (integer_op >> 1).astype(bool)
    binary_z = np.mod(integer_op, 2).astype(bool)

    np.concatenate((binary_x, binary_z), axis=1, out=binary)

    return binary


def integer_to_qubit_terms(integer_op, factors):
    """Perform conversion from integer array to qubit terms.

    Returns:
        dict: Pauli terms and coefficient.
    """

    qubit_operator = QubitOperator()
    for n_term, term in enumerate(integer_op):
        # Convert integer to a qubit terms (e.g. [0 1 2 3] => ((1, "Z"),
        # (2, "X"), (3, "Y"))).
        tuple_term = tuple([(qubit_i, ConvertPauli(int(term[qubit_i])).char) for
                             qubit_i in range(len(term)) if int(term[qubit_i]) > 0])
        # If a term is the same, QubitOperator takes into account the summation.
        qubit_operator += QubitOperator(tuple_term, factors[n_term])

    return qubit_operator.terms


def is_commuting(hybrid_op_a, hybrid_op_b, term_resolved=False):
    """Check if two operators commute, using stabilizer model. With the
    *term_resolved* flag, the user can identify which terms in the parent
    operator do or do not commute with the target operator. In stabilizer
    notation, (ax|az)(bz|bx) = -1^(ax.bz + az.bx) (bz|bx)(ax|az) define the
    commutation relations. By evaluating sign of (ax.bz + az.bx), we recover
    commutation relation. For a given pair of terms in the operators,
    ax.bz + az.bx = sum(XOR (AND (a' . b))) where a' is the z-x swapped
    a-vector. We then apply an OR reduction over all of the b-terms to
    identify whether term a_i commutes with all of b_j. Applying once again,
    an OR reduction over all a_i, we get the commutator for the entirety of
    a,b.

    Args:
        other_operator: other HybridOperator.
        term_resolved (bool): If True, get commutator for each term.

    Returns:
        bool or numpy array of bool: If operators commutes or array of bool
            describing which terms are commuting.
    """

    term_bool = np.zeros(hybrid_op_a.n_terms, dtype=bool)

    # Binary_swap is used to check the commuation relation.
    for index, term in enumerate(hybrid_op_a.binary_swap):
        term_bool[index] = np.logical_or.reduce(
            np.logical_xor.reduce(np.bitwise_and(term, hybrid_op_b.binary),
            axis=1))

    if not term_resolved:
        return not np.logical_or.reduce(term_bool)
    else:
        return np.logical_not(term_bool)
