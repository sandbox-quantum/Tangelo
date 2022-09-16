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
in order to facilitate post-processing of quantum experiments
"""

from collections import Counter

epsilon = 1e-2


class Histogram:

    def __init__(self, freqs, n_shots):

        self._freqs = freqs.copy()
        self._n_shots = n_shots

        # Flags histograms whose frequencies do not add to 1. (more than epsilon % of shots removed)
        sum_values = sum(self.freqs.values())
        if sum_values < 1 - epsilon:
            raise Warning(f"Histogram frequencies not summing very close to 1. (sum = {sum_values}) ")

        # Error: bitstrings not of consistent length
        lengths_bitstrings = set([len(k) for k in self.freqs])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f"Entries in frequency dictionary not of consistent length. Please check your input.")

    def __eq__(self, other):
        """ Two histograms are equal if they have same frequencies and number of shots """
        return (self.freqs == other.freqs) and (self.n_shots == other.n_shots)

    @property
    def freqs(self):
        return self._freqs

    @property
    def n_shots(self):
        return self._n_shots


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
        # Check that freq dicts have same bitstring length
        lengths_bitstrings = set([len(next(iter(h.freqs))) for h in hists])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f'Input histograms have different bitstring lengths ({lengths_bitstrings})')

        # Aggregate histograms
        total_counter = sum([Counter({k: round(v*h.n_shots) for k, v in h.freqs.items()}) for h in hists],
                            start=Counter())
        total_shots = sum(h.n_shots for h in hists)
        return Histogram({k: v/total_shots for k, v in dict(total_counter).items()}, total_shots)
