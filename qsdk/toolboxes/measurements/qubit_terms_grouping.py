"""
    This file provides access to different approaches aiming at reducing the number of measurements required
    to compute the expectation value of a qubit (Pauli) operator. They attempt to identify the minimal amount of
    measurement bases / terms required to deduce the expectation values associated to the other terms present in the
    Pauli operator.
"""

import warnings
from openfermion.measurements import group_into_tensor_product_basis_sets
from qsdk.backendbuddy import Simulator


def group_qwc(qb_ham, seed=None):
    """
        Wrapper around Openfermion functionality that takes as input a QubitOperator and yields a collection of
        mesurement bases defining a partition of groups of sub-operators with terms that are diagonal in the same tensor
        product basis. Each sub-operator can be measured using the same qubit post-rotations in expectation estimation.
        This uses the idea of qubitwise commutativity (qwc).

        The resulting dictionary maps the measurement basis (key) to the list of qubit operators whose expectation
        value can be computed using the corresponding circuit. The size of this dictionary determines how many
        quantum circuits need to be executed in order to provide the expectation value of the input qubit operator.

        Args:
            operator (QubitOperator): the operator that will be split into sub-operators (tensor product basis sets).
            seed (int): default None. Random seed used to initialize the numpy.RandomState pseudo-RNG.
        Returns:
            sub_operators (dict): a dictionary where each key defines a tensor product basis,
            and each corresponding value is a QubitOperator with terms that are all diagonal in that basis.
    """

    return group_into_tensor_product_basis_sets(qb_ham, seed)


def exp_value_from_measurement_bases(sub_ops, histograms):
    """
        Computes the expectation value of the sum of all suboperators corresponding to the different measurement bases
        in the input dictionary. This is how one would go about computing the expectation value of an operator
        while trying to send a limited amount of tasks to a quantum computer, thereby lowering the cost in terms of
        number of measurements needed.

        Args:
            sub_ops (dict): Maps measurement bases to suboperators whose exp. value can be computed from it
            histograms (dict): Maps measurement bases to histograms corresponding to that quantum circuit
        Returns:
            exp_value (float or complex): Expectation value of the sum of all suboperators
    """

    # Warning if dicts dont have exact set of keys
    if set(sub_ops) != set(histograms):
        warnings.warn("Measurement bases are not exactly the same in both dictionaries. Terms may be missing.")

    exp_value = 0.
    for basis, freqs in histograms.items():
        for term, coef in sub_ops[basis].terms.items():
            exp_value += Simulator.get_expectation_value_from_frequencies_oneterm(term, freqs) * coef

    return exp_value
