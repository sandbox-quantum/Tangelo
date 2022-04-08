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

"""This file provides access to different approaches aiming at reducing the
number of measurements required to compute the expectation value of a qubit
(Pauli) operator. They attempt to identify the minimal amount of measurement
bases / terms required to deduce the expectation values associated to the other
terms present in the Pauli operator.
"""

import warnings
from openfermion.measurements import group_into_tensor_product_basis_sets
from tangelo.linq import Simulator


def group_qwc(qb_ham, seed=None):
    """Wrapper around Openfermion functionality that takes as input a
    QubitOperator and yields a collection of mesurement bases defining a
    partition of groups of sub-operators with terms that are diagonal in the
    same tensor product basis. Each sub-operator can be measured using the same
    qubit post-rotations in expectation estimation. This uses the idea of
    qubitwise commutativity (qwc).

    The resulting dictionary maps the measurement basis (key) to the list of
    qubit operators whose expectation value can be computed using the
    corresponding circuit. The size of this dictionary determines how many
    quantum circuits need to be executed in order to provide the expectation
    value of the input qubit operator.

    Args:
        operator (QubitOperator): the operator that will be split into
            sub-operators (tensor product basis sets).
        seed (int): default None. Random seed used to initialize the
            numpy.RandomState pseudo-RNG.

    Returns:
        dict: a dictionary where each key defines a tensor product basis, and
            each corresponding value is a QubitOperator with terms that are all
            diagonal in that basis.
    """

    return group_into_tensor_product_basis_sets(qb_ham, seed)


def group_sorted_qwc(op):
    """
    Partitioning function that identifies measurement bases and maps them to qubit operators that can be measured
    with it. This partitioning implements the sorted insertion algorithm, and is deterministic.

    Args:
        op (QubitOperator): Input qubit operator

    Returns:
        dict: dictionary mapping measurement bases to a corresponding qubit operator
    """

    # Sort terms in decreasing magnitude for coefficients.
    terms = [(k, v) for k, v in sorted(op.terms.items(), key=lambda x: abs(x[1]), reverse=True)]

    # Builds measurement bases gradually, grouping terms with largest coefficients when they commute qubitwise
    sorted_qwc_groups = dict()
    for pauli_word, coeff in terms:
        commutes = False
        for k, v in sorted_qwc_groups.items():
            if do_bases_commute_qubitwise(pauli_word, k):
                commutes = True
                basis = tuple(set(k) | set(pauli_word))
                q = QubitOperator(); q.terms = {pauli_word: coeff}
                del sorted_qwc_groups[k]
                sorted_qwc_groups[basis] = v + q
                break
        if not commutes:
            q = QubitOperator(); q.terms = {pauli_word: coeff}
            sorted_qwc_groups[pauli_word] = q

    return sorted_qwc_groups


def do_bases_commute_qubitwise(b1, b2):
    """ Helper function. Checks whether two bases b1 and b2 commute qubitwise. """
    if not (b1 and b2):
        return False

    b1_dict, b2_dict = dict(b1), dict(b2)
    for i in set(b1_dict) & set(b2_dict):
        if b1_dict[i] != b2_dict[i]:
            return False
    return True


def exp_value_from_measurement_bases(sub_ops, histograms):
    """Computes the expectation value of the sum of all suboperators
    corresponding to the different measurement bases in the input dictionary.
    This is how one would go about computing the expectation value of an
    operator while trying to send a limited amount of tasks to a quantum
    computer, thereby lowering the cost in terms of number of measurements
    needed.

    Args:
        sub_ops (dict): Maps measurement bases to suboperators whose exp. value
            can be computed from it.
        histograms (dict): Maps measurement bases to histograms corresponding to
            that quantum circuit.

    Returns:
        float or complex: Expectation value of the sum of all suboperators.
    """

    # Warning if dicts do not have same exact set of keys
    if set(sub_ops) != set(histograms):
        warnings.warn("Measurement bases are not exactly the same in both dictionaries. Terms may be missing.")

    exp_value = 0.
    for basis, freqs in histograms.items():
        for term, coef in sub_ops[basis].terms.items():
            exp_value += Simulator.get_expectation_value_from_frequencies_oneterm(term, freqs) * coef

    return exp_value
