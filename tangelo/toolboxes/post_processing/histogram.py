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
    """Class to provide useful tools helping redundant tasks when analyzing
    data from an experiment. The expected data input is an histogram of
    bistrings ("010..."), where 0/1 correspond to the |0>/|1> measured state in
    the computational basis. The outcomes can refer to either the number of
    replicas (where the summation across the bistrings equals the number of
    shots), or the probability (where the summation equals to 1). In the former
    case, the number of shots must be provided.

    The internal representation is kept as an histogram of shots. Normalized
    quantities are accessible via the Histogram "frequencies" properties.

    Args:
        outcomes (dict): Results in the format of bistring: outcome, where
            outcome can be an int (number of replicas) or a float (probability).
        n_shots (int): Self-explanatory. If it is greater than 0, the class
            considers that probabilities are provided.
        msq_first (bool): Bit ordering. For example, 011 (msq_first) = 110
            (lsq_first). This depends on the hardware provider convention.
        epsilon (float): When probabilities are provided, this parameter is used
            to check if the sum is within [1-epsilon, 1+epsilon]. Default is
            1e-2.
    """

    def __init__(self, outcomes, n_shots=0, msq_first=False, epsilon=1e-2):

        # Error: bitstrings not of consistent length.
        lengths_bitstrings = set([len(k) for k in outcomes.keys()])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f"Entries in outcomes dictionary not of consistent length. Please check your input.")

        if n_shots > 0:
            # Flags histograms whose frequencies do not add to 1.
            sum_values = sum(outcomes.values())
            if abs(sum_values-1) > epsilon:
                raise ValueError(f"Histogram frequencies not summing very close to 1. (sum = {sum_values}).")

            outcomes = {k: round(v*n_shots) for k, v in outcomes.items()}

        self.counts = outcomes.copy()

        if msq_first:
            self.counts = {k[::-1]: v for k, v in self.counts.items()}

    def __repr__(self):
        """Output the string representation."""
        return f"{self.counts}"

    def __eq__(self, other):
        """Two histograms are equal if they have same counts, i.e. same
        bistrings and outcomes.
        """
        return (self.counts == other.counts)

    def __add__(self, other):
        """The aggregate_histograms function is used to add two Histogram objects."""
        return aggregate_histograms(self, other)

    def __iadd__(self, other):
        new_histogram = self + other
        self.__dict__ = new_histogram.__dict__
        return self

    @property
    def n_shots(self):
        """Return the number of shots encoded in the Histogram.

        Returns:
            int: Self-explanatory.
        """
        return sum(self.counts.values())

    @property
    def n_qubits(self):
        """The length of the bistrings represents the number of qubits."

        Returns:
            int: Self-explanatory.
        """
        return len(list(self.counts.keys())[0])

    @property
    def frequencies(self):
        """The frequencies are normalized counts vs the number of shots.

        Returns:
            dict: Frequencies in a {bistring: probability, ...} format.
        """
        return {bistring: counts/self.n_shots for bistring, counts in self.counts.items()}

    def post_select(self, values):
        """Post selection is done with the post_select function (see the
        relevant documentation for more details).
        """
        new_hist = post_select(self, values, return_only_counts=False)
        self.__dict__ = new_hist.__dict__

    def resample(self, n_shots):
        """Post selection is done with the tangelo.toolboxes.post_processing.boostrapping.get_resampled_frequencies
        function (see the relevant documentation for more details).

        Returns:
            Histogram: resampled data with n_shots from the distribution.
        """
        new_frequencies = get_resampled_frequencies(self.frequencies, n_shots)
        return Histogram(new_frequencies, n_shots)

    def get_expectation_value(self, term, coeff=1.):
        """Output the expectation value for qubit operator term. The histogram
        data is expected to be results from a circuit with the proper qubit
        rotations.
        """
        return coeff*Simulator.get_expectation_value_from_frequencies_oneterm(term, self.frequencies)


def aggregate_histograms(*hists):
    """Aggregate all input Histogram objects together.

    Args:
        hists (variable number of Histogram objects): the input Histogram objects.

    Returns:
        Histogram: The aggregated histogram.
    """

    if not hists:
        raise ValueError(f"{aggregate_histograms.__name__} takes at least one Histogram object as an argument")
    elif len(hists) == 1:
        return hists[0]
    else:
        # Check that data is on the same number of qubits (same bitstring lengths).
        lengths_bitstrings = set([h.n_qubits for h in hists])
        if len(lengths_bitstrings) > 1:
            raise ValueError(f"Input histograms have different bitstring lengths ({lengths_bitstrings})")

        # Aggregate histograms.
        total_counter = sum([Counter({k: v for k, v in h.counts.items()}) for h in hists],
                            start=Counter())

        return Histogram(dict(total_counter))

def post_select(hist, expected_outcomes):
    """Post selection function to select data when a supplementary circuit is
    appended to the quantum state. Symmetry breaking results are rejected and
    the new Histogram oject has less qubits than the original results (depending
    on how many symmetries are checked with ancilla qubits).

    Args:
        hist (Histogram): Self-explanatory.
        expected_outcomes (dict): Wanted outcomes on certain qubit indices and
            their expected state. For example, {0: "1"} would filter results
            based on the first qubit with the 1 state measured. This argument
            can also filter many qubits.

    Returns:
        Histogram: New Histogram with filtered outcomes based on expected
            outcomes.

    """

    new_counts = {}
    for bitstring, counts in hist.counts.items():
        if all([bitstring[qubit_i] == bit for qubit_i, bit in expected_outcomes.items()]):
            new_bistring = "".join([bitstring[qubit_i] for qubit_i in range(len(bitstring)) if qubit_i not in expected_outcomes.keys()])
            new_counts[new_bistring] = counts

    return Histogram(new_counts, msq_first=False)
