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

""" This module provides a Histogram class and functions to manipulate them,
in order to facilitate post-processing of quantum experiments.
"""

from collections import Counter

from tangelo.linq import Simulator
from tangelo.toolboxes.post_processing.bootstrapping import get_resampled_frequencies


class Histogram:
    """
    Ordering lsq, |000...> are qubits 0, 1, 2, ...
    """

    def __init__(self, outcomes, n_shots=0, msq_first=False, epsilon=1e-2):

        # Error: bitstrings not of consistent length
        lengths_bitstrings = set([len(k) for k in outcomes.keys()])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f"Entries in outcomes dictionary not of consistent length. Please check your input.")

        if n_shots > 0:
            # Flags histograms whose frequencies do not add to 1. (more than epsilon % of shots removed)
            sum_values = sum(outcomes.values())
            if abs(sum_values-1) > epsilon:
                raise Warning(f"Histogram frequencies not summing very close to 1. (sum = {sum_values}).")

            outcomes = {k: round(v*n_shots) for k, v in outcomes}

        self.counts = outcomes.copy()

        if msq_first:
            self.counts = {k[::-1]: v for k, v in self.counts.items()}

    def __repr__(self):
        return f"{self.counts}"

    @property
    def n_shots(self):
        return sum(self.counts.values())

    @property
    def n_qubits(self):
        return len(list(self.counts.keys())[0])

    @property
    def frequencies(self):
        return {bistring: counts/self.n_shots for bistring, counts in self.counts.items()}

    def post_select(self, values):
        new_hist = post_select(self, values, return_only_counts=False)
        self.__dict__ = new_hist.__dict__

    def resample(self, n_shots):
        new_frequencies = get_resampled_frequencies(self.frequencies, n_shots)
        return Histogram(new_frequencies, n_shots)

    def get_expectation_value(self, qubit_operator):

        expectation = 0.
        for term, coeff in qubit_operator.terms.items():
            expectation += coeff*Simulator.get_expectation_value_from_frequencies_oneterm(term, self.frequencies)

        return expectation


def aggregate_histograms(*hists):
    """ Aggregate all input histograms together.

    Args:
        hists (variable number of Histogram objects): the input histograms

    Returns:
        Histogram: The aggregated histogram.
    """

    if not hists:
        raise ValueError(f"{aggregate_histograms.__name__} takes at least one Histogram object as an argument")

    elif len(hists) == 1:
        return hists[0]

    else:
        # Check that freq dicts have same bitstring lengths.
        lengths_bitstrings = set([h.n_qubits for h in hists])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f'Input histograms have different bitstring lengths ({lengths_bitstrings})')

        # Aggregate histograms.
        total_counter = sum([Counter({k: v for k, v in h}) for h in hists],
                            start=Counter())

        return Histogram(dict(total_counter))

def compare_frequencies(hist_a, hist_b, print_diff=False):
    return

def post_select(hist, values, return_only_counts=False):

    if isinstance(hist, Histogram):
        hist = hist.counts

    new_counts = {}
    for bitstring, counts in hist.items():
        if all([bitstring[qubit_i] == bit for qubit_i, bit in values.items()]):
            new_bistring = "".join([bitstring[qubit_i] for qubit_i in range(len(bitstring)) if qubit_i not in values.keys()])
            new_counts[new_bistring] = counts

    return new_counts if return_only_counts else Histogram(new_counts, msq_first=False)
