"""
    This file provides functions allowing users to estimate the number of measurements that are needed to
    approximate the expectation values of some qubit operators up to a given accuracy.

    The functions range from simple general heuristics to more complex approaches that may leverage our knowledge
    of the quantum state.
"""

import math


def get_measurement_estimate(qb_ham, digits=3, method="uniform"):
    """ Given a qubit operator and a level of accuracy, computes the number of measurements required by each term
     of the qubit operator to reach the accuracy provided by the user when computing expectation values, returns it
     as a dictionary mapping measurement basis to number of measurements.

     "uniform" method makes no assumption about the underlying probability distribution resulting from the quantum
     state preparation circuit. The rule of thumb is "Multiply number of samples by 100 for each digit of accuracy".
     """

    available_methods = {'uniform'}
    if method not in available_methods:
        raise NotImplementedError(f"Only available methods are {available_methods}")

    scaling_factor = 100**(digits+1)

    measurements = dict()
    for term, coef in qb_ham.terms.items():
        coef = max(abs(coef.real), abs(coef.imag))
        if not term or coef < 10**(-digits):
            measurements[term] = 0
        else:
            n_samples = math.floor(scaling_factor * 100**(math.log10(coef)))
            # Assign to dictionary, handle edge case
            measurements[term] = 10 if n_samples == 1 else n_samples

    return measurements
