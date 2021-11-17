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

"""This file provides functions allowing users to estimate the number of
measurements that are needed to approximate the expectation values of some qubit
operators up to a given accuracy.

The functions range from simple general heuristics to more complex approaches
that may leverage our knowledge of the quantum state.
"""

import math


def get_measurement_estimate(qb_op, digits=3, method="uniform"):
    """Given a qubit operator and a level of accuracy, computes the number of
    measurements required by each term of the qubit operator to reach the
    accuracy provided by the user when computing expectation values, returns it
    as a dictionary mapping measurement basis to number of measurements.

    "uniform" method makes no assumption about the underlying probability
    distribution resulting from the quantum state preparation circuit. The rule
    of thumb is "Multiply number of samples by 100 for each digit of accuracy".

    Args:
        qb_op: qubit operator.
        digits (integer): number of digits of accuracy desired on expectation
            value.

    Returns:
        dict: Dictionary mapping terms / measurement bases to their number of
            measurements.
    """

    available_methods = {'uniform'}
    if method not in available_methods:
        raise NotImplementedError(f"Only available methods are {available_methods}")

    scaling_factor = 100**(digits+1)

    measurements = dict()
    for term, coef in qb_op.terms.items():
        coef = max(abs(coef.real), abs(coef.imag))
        if not term or coef < 10**(-digits):
            measurements[term] = 0
        else:
            n_samples = math.floor(scaling_factor * 100**(math.log10(coef)))
            # Assign to dictionary, handle edge case
            measurements[term] = 100 if n_samples == 1 else n_samples

    return measurements
