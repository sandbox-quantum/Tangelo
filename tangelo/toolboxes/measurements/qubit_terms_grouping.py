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


def group_qwc(qb_ham, seed=None, n_repeat=1):
    """Wrapper around Openfermion functionality that takes as input a
    QubitOperator and yields a collection of measurement bases defining a
    partition of groups of sub-operators with terms that are diagonal in the
    same tensor product basis. Each sub-operator can be measured using the same
    qubit post-rotations in expectation estimation. This uses the idea of
    qubitwise commutativity (qwc), and the minimum clique cover algorithm.

    The resulting dictionary maps the measurement basis (key) to the list of
    qubit operators whose expectation value can be computed using the
    corresponding circuit. The size of this dictionary determines how many
    quantum circuits need to be executed in order to provide the expectation
    value of the input qubit operator.

    The minimum clique cover algorithm can be initialized with a random seed
    and can be repeated several times with different seeds in order to return
    the run that produces the lowest number of groups.

    Args:
        qb_ham (QubitOperator): the operator that will be split into
            sub-operators (tensor product basis sets).
        seed (int): default None. Random seed used to initialize the
            numpy.RandomState pseudo-RNG.
        n_repeat (int): Repeat with a different random seed, keep the outcome
            resulting in the lowest number of groups.

    Returns:
        dict: a dictionary where each key defines a tensor product basis, and
            each corresponding value is a QubitOperator with terms that are all
            diagonal in that basis.
    """

    res = group_into_tensor_product_basis_sets(qb_ham, seed)
    for i in range(n_repeat-1):
        res2 = group_into_tensor_product_basis_sets(qb_ham)
        if len(res2) < len(res):
            res = res2
    return res


def check_bases_commute_qwc(b1, b2):
    """ Check whether two bases commute qubitwise.

    Args:
        b1 (tuple of (int, str)): the first measurement basis
        b2 (tuple of (int, str)): the second measurement basis

    Returns:
        bool: whether or not the bases commute qubitwise
    """

    b1_dict, b2_dict = dict(b1), dict(b2)
    for i in set(b1_dict) & set(b2_dict):
        if b1_dict[i] != b2_dict[i]:
            return False
    return True


def map_measurements_qwc(qwc_group_map):
    """ Somewhat reverses a grouping dictionary, linking all measurement bases to the list of
    group representatives that commute with them qubitwise. Useful to find all the bases
    whose shots can be used in order to form data for other bases during a hardware experiment.

    Args:
        qwc_group_map (dict): maps a measurement basis to a qubit operator whose terms commute
            with said measurement basis qubit-wise.
    Returns:
        dict: maps each of the bases to all group representatives that commute with them qubitwise.
    """

    meas_map = dict()
    meas_bases, op_bases = list(), list()
    for k, v in qwc_group_map.items():
        meas_bases.append(k)
        op_bases.extend(v.terms.keys())

    for b1 in op_bases:
        for b2 in meas_bases:
            if b1 and check_bases_commute_qwc(b1, b2):
                meas_map[b1] = meas_map.get(b1, []) + [b2]

    return meas_map


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

    # Warning if dicts dont have exact set of keys
    if set(sub_ops) != set(histograms):
        warnings.warn("Measurement bases are not exactly the same in both dictionaries. Terms may be missing.")

    exp_value = 0.
    for basis, freqs in histograms.items():
        for term, coef in sub_ops[basis].terms.items():
            exp_value += Simulator.get_expectation_value_from_frequencies_oneterm(term, freqs) * coef

    return exp_value
