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

"""This module defines various kinds of operators used in vqe. It can later be
broken down in several modules if needed.
"""

from math import sqrt
from collections import OrderedDict

# Later on, if needed, we can extract the code for the operators themselves to remove the dependencies and customize
import openfermion


class FermionOperator(openfermion.FermionOperator):
    """Currently, this class is coming from openfermion. Can be later on be
    replaced by our own implementation.
    """
    pass


class QubitOperator(openfermion.QubitOperator):
    """Currently, this class is coming from openfermion. Can be later on be
    replaced by our own implementation.
    """

    def frobenius_norm_compression(self, epsilon, n_qubits):
        """Reduces the number of operator terms based on its Frobenius norm
        and a user-defined threshold, epsilon. The eigenspectrum of the
        compressed operator will not deviate more than epsilon. For more
        details, see J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.

        Args:
            epsilon (float): Parameter controlling the degree of compression
                and resulting accuracy.
            n_qubits (int): Number of qubits in the register.

        Returns:
            QubitOperator: The compressed qubit operator.
        """

        compressed_op = dict()
        coef2_sum = 0.
        frob_factor = 2**(n_qubits // 2)

        # Arrange the terms of the qubit operator in ascending order
        self.terms = OrderedDict(sorted(self.terms.items(), key=lambda x: abs(x[1]), reverse=False))

        for term, coef in self.terms.items():
            coef2_sum += abs(coef)**2
            # while the sum is less than epsilon / factor, discard the terms
            if sqrt(coef2_sum) > epsilon / frob_factor:
                compressed_op[term] = coef
        self.terms = compressed_op
        self.compress()


class QubitHamiltonian(QubitOperator):
    """QubitHamiltonian objects are essentially openfermion.QubitOperator
    objects, with extra attributes. The mapping procedure (mapping) and the
    qubit ordering (up_then_down) are incorporated into the class. In addition
    to QubitOperator, several checks are done when performing arithmetic
    operations on QubitHamiltonians.

    Attributes:
        term (openfermion-like): Same as openfermion term formats.
        coefficient (complex): Coefficient for this term.
        mapping (string): Mapping procedure for fermionic to qubit encoding
            (ex: "JW", "BK", etc.).
        up_then_down (bool): Whether or not spin ordering is all up then
            all down.

    Properties:
        n_terms (int): Number of terms in this qubit Hamiltonian.
    """

    def __init__(self, term=None, coefficient=1., mapping=None, up_then_down=None):
        super(QubitOperator, self).__init__(term, coefficient)

        self.mapping = mapping
        self.up_then_down = up_then_down

    @property
    def n_terms(self):
        return len(self.terms)

    def __iadd__(self, other_hamiltonian):

        # Raise error if attributes are not the same across Hamiltonians. This
        # check is ignored if comparing to a QubitOperator or a bare
        # QubitHamiltonian.
        if self.mapping is not None and self.up_then_down is not None and \
                                other_hamiltonian.mapping is not None and \
                                other_hamiltonian.up_then_down is not None:

            if self.mapping.upper() != other_hamiltonian.mapping.upper():
                raise RuntimeError("Mapping must be the same for all QubitHamiltonians.")
            elif self.up_then_down != other_hamiltonian.up_then_down:
                raise RuntimeError("Spin ordering must be the same for all QubitHamiltonians.")

        return super(QubitOperator, self).__iadd__(other_hamiltonian)

    def __eq__(self, other_hamiltonian):

        # Additional checks for == operator. This check is ignored if comparing
        # to a QubitOperator or a bare QubitHamiltonian.
        if self.mapping is not None and self.up_then_down is not None and \
                                other_hamiltonian.mapping is not None and \
                                other_hamiltonian.up_then_down is not None:
            if (self.mapping.upper() != other_hamiltonian.mapping.upper()) or (self.up_then_down != other_hamiltonian.up_then_down):
                return False

        return super(QubitOperator, self).__eq__(other_hamiltonian)

    def to_qubitoperator(self):
        qubit_op = QubitOperator()
        qubit_op.terms = self.terms.copy()

        return qubit_op


def count_qubits(qb_op):
    """Return the number of qubits used by the qubit operator based on the
    highest index found in the terms.
    """
    if (len(qb_op.terms.keys()) == 0) or ((len(qb_op.terms.keys()) == 1) and (len(list(qb_op.terms.keys())[0]) == 0)):
        return 0
    else:
        return max([(sorted(pw))[-1][0] for pw in qb_op.terms.keys() if len(pw) > 0]) + 1


def normal_ordered(fe_op):
    """ Input: a Fermionic operator of class
    toolboxes.operators.FermionicOperator or openfermion.FermionicOperator for
    reordering.

    Returns:
        FermionicOperator: Normal ordered operator.
    """

    # Obtain normal ordered fermionic operator as list of terms
    norm_ord_terms = openfermion.transforms.normal_ordered(fe_op).terms

    # Regeneratore full operator using class of tangelo.toolboxes.operators.FermionicOperator
    norm_ord_fe_op = FermionOperator()
    for term in norm_ord_terms:
        norm_ord_fe_op += FermionOperator(term, norm_ord_terms[term])
    return norm_ord_fe_op


def squared_normal_ordered(all_terms):
    """Input: a list of terms to generate toolboxes.operators.FermionOperator
    or openfermion.FermionOperator

    Returns:
        FermionOperator: squared (i.e. fe_op*fe_op) and
            normal ordered.
    """

    # Obtain normal ordered fermionic operator as list of terms
    fe_op = list_to_fermionoperator(all_terms)
    fe_op *= fe_op
    return normal_ordered(fe_op)


def list_to_fermionoperator(all_terms):
    """Input: a list of terms to generate FermionOperator

    Returns:
        FermionOperator: Single merged operator.
    """

    fe_op = FermionOperator()
    for item in all_terms:
        fe_op += FermionOperator(item[0], item[1])
    return fe_op


def qubitop_to_qubitham(qubit_op, mapping, up_then_down):
    """Function to convert a QubitOperator into a QubitHamiltonian.

    Args:
        qubit_op (QubitOperator): Self-explanatory.
        mapping (string): Qubit mapping procedure.
        up_then_down (bool): Whether or not spin ordering is all up then
            all down.

    Returns:
        QubitHamiltonian: Self-explanatory.
    """
    qubit_ham = QubitHamiltonian(mapping=mapping, up_then_down=up_then_down)
    qubit_ham.terms = qubit_op.terms.copy()

    return qubit_ham
