# Copyright 2023 Good Chemistry Company.
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

from tangelo.linq import get_expectation_value_from_frequencies_oneterm
from tangelo.toolboxes.post_processing.bootstrapping import get_resampled_frequencies


class Histogram:
    """Class to provide useful tools helping redundant tasks when analyzing
    data from an experiment. The expected data input is an histogram of
    bitstrings ("010..."), where 0 (resp. 1) correspond to the |0> (resp. |1>)
    measured state in the computational basis. The outcomes can refer to either
    the shot counts for each computational basis, or the corresponding
    probability.

    The internal representation is kept as an histogram of shots. Normalized
    quantities are accessible via the Histogram "frequencies" properties.
    Bitstrings are stored with the lowest-significant qubit first (lsq_first)
    ordering.

    Args:
        outcomes (dict of string: int or float): Results in the format of bitstring:
        outcome, where outcome can be a number of shots a probability.
        n_shots (int): Self-explanatory. If it is equal 0, the class
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
                raise ValueError(f"Sum of Histogram frequencies ({sum_values}) differ from 1. by more than an epsilon ({epsilon})."
                    "Please adjust the value of epsilon or adjust shot data.")

            outcomes = {k: round(v*n_shots) for k, v in outcomes.items()}
        elif n_shots < 0:
            raise ValueError(f"The number of shots provided ({n_shots}) must be a positive value.")

        self.counts = outcomes.copy()

        if msq_first:
            self.counts = {k[::-1]: v for k, v in self.counts.items()}

    def __repr__(self):
        """Output the string representation."""
        return f"{self.counts}"

    def __eq__(self, other):
        """Two histograms are equal if they have same counts, i.e. same
        bitstrings and outcomes.
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
        """The length of the bitstrings represents the number of qubits."

        Returns:
            int: Self-explanatory.
        """
        return len(next(iter(self.counts)))

    @property
    def frequencies(self):
        """The frequencies are normalized counts vs the number of shots.

        Returns:
            dict: Frequencies in a {bitstring: probability, ...} format.
        """
        return {bitstring: counts/self.n_shots for bitstring, counts in self.counts.items()}

    def remove_qubit_indices(self, *indices):
        """Method to remove qubit indices in the result. The remaining
        bitstrings (of new length = length - N_indices) are summed up together.

        Args:
            indices (variable number of int): Qubit indices to remove.
        """
        new_counts = dict()
        for bitstring, counts in self.counts.items():
            new_bitstring = "".join([bitstring[qubit_i] for qubit_i in range(len(bitstring)) if qubit_i not in indices])
            new_counts[new_bitstring] = new_counts.get(new_bitstring, 0) + counts

        self.counts = new_counts

    def post_select(self, expected_outcomes):
        """Method to apply post selection on Histogram data, based on a post
        selection function in this module. This method changes the Histogram
        object inplace. This is explicitly done on desired outcomes for specific
        qubit indices.

        Args:
            expected_outcomes (dict): Desired outcomes on certain qubit indices
                and their expected state. For example, {0: "1", 1: "0"} would
                filter results based on the first qubit with the |1> state and
                the second qubit with the |0> state measured.
        """
        def f_post_select(bitstring):
            for qubit_i, expected_bit in expected_outcomes.items():
                if bitstring[qubit_i] != expected_bit:
                    return False
            return True

        new_hist = filter_hist(self, f_post_select)
        self.counts = new_hist.counts
        self.remove_qubit_indices(*list(expected_outcomes.keys()))

    def resample(self, n_shots):
        """Generating new Histogram with n_shots, done with the
        get_resampled_frequencies function (see the relevant documentation for
        more details).

        Args:
            n_shots (int): Self-explanatory.

        Returns:
            Histogram: resampled data with n_shots from the distribution.
        """
        new_frequencies = get_resampled_frequencies(self.frequencies, n_shots)
        return Histogram(new_frequencies, n_shots)

    def get_expectation_value(self, term, coeff=1.):
        """Output the expectation value for qubit operator term. The histogram
        data is expected to have been acquired in a compatible measurement
        basis.

        Args:
            term(openfermion-style QubitOperator object): a qubit operator, with
                only a single term.
            coeff (imaginary): Coefficient to multiply the eigenvalue by.

        Returns:
            imaginary: Expectation value for this operator.
        """
        return coeff*get_expectation_value_from_frequencies_oneterm(term, self.frequencies)


def aggregate_histograms(*hists):
    """Return a Histogram object formed from data aggregated from all input
    Histogram objects.

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
        total_counter = sum([Counter({k: v for k, v in h.counts.items()}) for h in hists], Counter())

        return Histogram(dict(total_counter))


def filter_hist(hist, function, *args, **kwargs):
    """Filter selection function to consider bitstrings in respect to a boolean
    predicate on a bitstring.

    Args:
        hist (Histogram): Self-explanatory.
        function (function): Given a bitstring and some arbitrary arguments, the
            predicate should return a boolean value.

    Returns:
        Histogram: New Histogram with filtered outcomes based on the predicate
            provided.

    """
    new_counts = {bitstring: counts for bitstring, counts in hist.counts.items() if function(bitstring, *args, **kwargs)}
    return Histogram(new_counts, msq_first=False)
