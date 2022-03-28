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
from tangelo.toolboxes.operators import QubitOperator


# Helper class to help converting from/to string <-> int <-> stabilizer.
class ConvertPauli:
    """Pauli convertion tool. It aims to replace multiple dictionaries.
    It performs the conversion of
    I <-> 0 <-> (0,0)
    Z <-> 1 <-> (0,1)
    X <-> 2 <-> (1,0)
    Y <-> 3 <-> (1,1)
    The input can be whatever found in the previous conversion table.
    Args:
        pauli_id (string, int or tuple of bool): Single Pauli.
    Attributes:
        char (string): Pauli letter.
        integer (int): Pauli integer.
        tuple (tuple of bool): Pauli binary representation.
    """

    def __init__(self, pauli_id):
        pauli_translation = [['I', 0, (0, 0)],
                             ['Z', 1, (0, 1)],
                             ['X', 2, (1, 0)],
                             ['Y', 3, (1, 1)]
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
            raise ValueError("Unknown Pauli identification..")


class HybridOperator():
    """Construct integer and binary representations of the operator, as based on
    the user-specified input. Internal representation is mostly done with an
    integer array. The coversion is defined as:
      Letter | Integer | Binary
    -    I   |    0    | (0,0)
    -    Z   |    1    | (0,1)
    -    X   |    2    | (1,0)
    -    Y   |    3    | (1,1)
    Most of the algorithm use the integer representation to detect symmetries
    and performed operation (ex: multiplication). At the end, the operator can be
    output as a QubitOperator.
    Args:
        n_qubits (int): Number of qubits in the problem.
        qubit_operator (QubitOperator): Qubit operator, based on openfermion.QubitOperator.
        integer_operator (array-like): Array of 0s, 1s, 2s and 3s. Each line
            represents a term, and each term is a Pauli word.
        binary_operator (array-like): Array of 0s (false) and 1s (true). Binary
            representation of the integer array.
        factors (array-like): Factors fopr each term. Without that, information
            would be lost when converting from a QubitOperator.
    """

    def __init__(self, n_qubits, qubit_operator=None, integer_operator=None, binary_operator=None, factors=None):
        self.n_qubits = n_qubits

        # TODO:
        # - Add restraints to put factors with the creation of HybridOperator.

        # Input is a QubitOperator.
        if qubit_operator is not None:
            self.factors = np.array([qubit_operator.terms[n_term] for n_term in qubit_operator.terms])
            self.n_terms = len(self.factors)
            self.integer = self.qubit_to_integer(qubit_operator)
            self.binary, self.binary_swap = self.integer_to_binary()
        # Input is a numpy array of 0s, 1s, 2s and 3s.
        elif integer_operator is not None:
            self.factors = factors
            self.n_terms = len(factors)
            self.integer = integer_operator
            self.binary, self.binary_swap = self.integer_to_binary()
        # Input is a numpy array of 0s and 1s.
        elif binary_operator is not None:
            self.factors = factors
            self.n_terms = len(factors)
            self.binary = binary_operator

            # Swapping the binary representation. Example: instead of (0,1), write (1,0).
            # This is useful in the future to detect commutation between terms.
            list_col_swap = [n_column for n_column in range(2 * n_qubits)]
            list_col_swap = list_col_swap[n_qubits:] + list_col_swap[:n_qubits]
            self.binary_swap = self.binary[:, list_col_swap]

            # The integer array is defined trivially with the binary array.
            self.integer = 2 * self.binary[:, :n_qubits] + self.binary[:, n_qubits:]
        else:
            raise ValueError("No operator has been defined!")

        # Attribute to be define when symmetries will be computed.
        self.kernel = None

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

        factors = np.zeros((self.n_terms*other_operator.n_terms), dtype=complex)
        product = np.zeros((len(factors), self.n_qubits), dtype=int)
        increment = other_operator.n_terms

        for n_term in range(self.n_terms):
            levi_term = levi[self.integer[n_term], other_operator.integer]
            non_zero = np.where(levi_term != 0)

            product[n_term * increment:(n_term+1) * increment] = non_zero[2].reshape((increment, self.n_qubits))
            factors[n_term * increment:(n_term+1) * increment] = self.factors[n_term] * other_operator.factors * np.product(levi_term[non_zero].reshape((increment, self.n_qubits)), axis=1)

        return HybridOperator(self.n_qubits, integer_operator=product, factors=factors)

    def qubit_to_integer(self, qubit_operator):
        """Perform conversion from qubit operator to integer array. The integer
        attribute is instantiated, and populated from the QubitOperator attribute.
        Args:
            qubit_operator (QubitOperator): Qubit operator with pauli words and
                factors.
        """

        integer = np.zeros((self.n_terms, self.n_qubits), dtype=int)

        for index, term in enumerate(qubit_operator.terms):
            for n_qubit, pauli_letter in term:
                integer[index, n_qubit] = ConvertPauli(pauli_letter).integer

        return integer

    def integer_to_binary(self):
        """Perform conversion from integer array to binary (stabilizer) array.
        Two arrays, binary, and binary_swap are defined, to facilitate
        identification of commutation relations.
        Returns:
            Tuple: Array of 0s and 1s + array of 1s and 0s representing the
                operator with the stabilizer notation.
        """

        binary = np.zeros((self.n_terms, 2*self.n_qubits), dtype=bool)
        binary_swap = np.zeros((self.n_terms, 2*self.n_qubits), dtype=bool)

        binary_x = (self.integer >> 1).astype(bool)
        binary_z = np.mod(self.integer, 2).astype(bool)

        np.concatenate((binary_x, binary_z) ,axis=1, out=binary)
        np.concatenate((binary_z, binary_x), axis=1, out=binary_swap)

        return binary, binary_swap

    def get_kernel(self):
        """Get the kernel for a matrix of binary integer. Identity matrix is
        appended, and the extended matrix is then reduced to column-echelon form.
        The null space is then extracted.
        Returns:
            Array-like: 2d numpy array of binary integer.
        """

        identity_to_append = np.identity(2*self.n_qubits)

        # Append identity to the matrix E.
        E_prime = np.concatenate((self.binary, identity_to_append), axis=0)

        # Put the matrix into column-echelon form.
        E_prime = E_prime.astype(bool)
        pivot = E_prime.shape[1] - 1
        active_rows = E_prime.shape[0] - E_prime.shape[1] - 1

        for row in range(active_rows, -1, -1):
            if E_prime[row, :pivot+1].max():
                indices = np.where(E_prime[row, :pivot+1])[0]

                if len(indices) > 1:
                    for i in range(1, len(indices)):
                        E_prime[:, indices[i]] = np.logical_xor(E_prime[:, indices[i]], E_prime[:, indices[0]])

                if len(indices) > 0:
                    E_prime[:, (indices[0], pivot)] = E_prime[:, (pivot, indices[0])]
                    pivot -= 1
        # End of putting the matrix into column-echelon form.

        # Extracting the kernel.
        kernel = np.array([E_prime[self.binary.shape[0]:, i] for i in range(self.n_qubits) if not E_prime[:self.binary.shape[0], i].max()])
        kernel = np.concatenate((kernel[:, self.n_qubits:], kernel[:,: self.n_qubits]), axis=1)

        self.kernel = kernel
        return kernel

    def commutes_with(self, other, term_resolved=False):
        """Check if two operators commute, using stabilizer model. With the
        *term_resolved* flag, the user can identify which terms in the parent
        operator do or do not commute with the target operator. In stabilizer
        notation, (ax|az)(bz|bx) = -1^(ax.bz + az.bx) (bz|bx)(ax|az) define the
        commutation relations. So by evaluating sign of (ax.bz + az.bx), we
        recover commutation relation. For a given pair of terms in the operators,
        ax.bz + az.bx = sum(XOR (AND (a' . b))) where a' is the z-x swapped a-vector.
        We then apply an OR reduction over all of the b-terms to identify
        whether term a_i commutes with all of b_j. Applying once again, an OR
        reduction over all a_i, we get the commutator for the entirety of a,b.
        Args:
            other: other HybridOperator.
            term_resolved (bool): If True, get commutator for each term.
        Returns:
            bool or numpy array of bool: If operators commutes or array of
                bool describing which terms are commuting.
        """

        term_bool = np.zeros(self.n_terms, dtype=bool)

        # Binary_swap is used to check the commuation relation.
        for index, term in enumerate(self.binary_swap):
            term_bool[index] = np.logical_or.reduce(np.logical_xor.reduce(np.bitwise_and(term, other.binary), axis=1))

        if not term_resolved:
            return not np.logical_or.reduce(term_bool)
        else:
            return np.logical_not(term_bool)

    def remove_terms(self, index):
        """Remove term(s) from operator. The associated entry is deleted from
        factors, integer, binary and binary_swap. The number of terms n_terms is
        decremented.
        Args:
            index (int): Remove all terms corresponding to the index.
        """

        if type(index) == int:
            index = np.array([index])

        self.factors = np.delete(self.factors, index, axis=0)
        self.integer = np.delete(self.integer, index, axis=0)
        self.binary = np.delete(self.binary, index, axis=0)
        self.binary_swap = np.delete(self.binary_swap, index, axis=0)
        self.n_terms -= len(index)

    def integer_to_qubit(self):
        """Utility function to create a QubitOperator from an integer array
        Pauli string.
        Returns:
            QubitOperator: Operator defined by a sum of factors and pauli words.
        """

        qubit_hamiltonian = QubitOperator()
        for n_term, term in enumerate(self.integer):
            tuples = tuple([(index, ConvertPauli(int(term[index])).char) for index in range(len(term)) if int(term[index]) > 0])
            qubit_hamiltonian += QubitOperator(tuples, self.factors[n_term])

        return qubit_hamiltonian

    def get_nontrivial(self):
        """Non trivial terms. """

        nonzero = np.nonzero(self.integer)
        _, breaks = np.unique(nonzero[0], return_index=True)
        np.append(breaks, len(nonzero[0] + 1))
        return nonzero, breaks
