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

import unittest

from tangelo.toolboxes.post_processing import Histogram, aggregate_histograms, filter_hist


class HistogramTest(unittest.TestCase):

    def test_init_histogram_from_counts(self):
        """Initialize histogram object with a dictionary of counts."""
        h = Histogram({"00": 60, "11": 40})

        self.assertEqual(h.counts, {"00": 60, "11": 40})
        self.assertEqual(h.n_shots, 100)
        self.assertEqual(h.n_qubits, 2)

        ref_freq = {"00": 0.6, "11": 0.4}
        for k, v in h.frequencies.items():
            self.assertAlmostEqual(v, ref_freq.get(k))

    def test_init_histogram_from_frequencies_and_nshots(self):
        """Initialize histogram object with a dictonary of probabilities and
        a number of shots.
        """

        d = {"00": 0.6, "11": 0.4}
        h = Histogram(d, n_shots=100)

        self.assertEqual(h.counts, {"00": 60, "11": 40})
        self.assertEqual(h.n_shots, 100)
        self.assertEqual(h.n_qubits, 2)

        ref_freq = {"00": 0.6, "11": 0.4}
        for k, v in h.frequencies.items():
            self.assertAlmostEqual(v, ref_freq.get(k))

    def test_init_histogram_negative_nshots(self):
        """Initialize histogram object with a negative n_shots number."""

        with self.assertRaises(ValueError):
            Histogram({"00": 0.6, "11": 0.4}, n_shots=-100)

    def test_histogram_inconsistent_bitstring_length(self):
        """Throw an error if bitstrings are not all of consistent length."""
        with self.assertRaises(ValueError):
            Histogram({"00": 60, "111": 40})

    def test_histogram_bitstring_ordering(self):
        """Test the behavior with different bit ordering."""
        h = Histogram({"110": 60, "001": 40}, msq_first=True)
        self.assertEqual(h.counts, {"011": 60, "100": 40})

    def test_histogram_bad_frequencies(self):
        """Test the behavior of the Histogram initialization with frequencies
        not summing up to 1.
        """
        with self.assertRaises(ValueError):
            Histogram({"00": 0.6, "11": 0.3}, n_shots=100, epsilon=1e-2)

        with self.assertRaises(ValueError):
            Histogram({"00": 0.6, "11": 0.5}, n_shots=100, epsilon=1e-2)

    def test_eq_two_histograms(self):
        """Test the == operator on two Histograms."""
        d1 = {"00": 60, "11": 40}
        d2 = {"00": 50, "11": 50}

        self.assertEqual(Histogram(d1), Histogram(d1))
        self.assertNotEqual(Histogram(d1), Histogram(d2))

    def test_adding_two_histograms(self):
        """Aggregate histograms into a new one."""

        h1 = Histogram({"00": 60, "11": 40})
        h2 = Histogram({"00": 60, "01": 40})

        hf = h1 + h2
        self.assertEqual(hf, Histogram({"00": 120, "11": 40, "01": 40}))

        hf += h1
        self.assertEqual(hf, Histogram({"00": 180, "11": 80, "01": 40}))

    def test_removing_one_qubit(self):
        """Test removing one bit in an Histogram."""

        h = Histogram({"00": 40, "01": 30, "10": 20, "11": 10})
        h.remove_qubit_indices(1)
        self.assertEqual(h, Histogram({"0": 70, "1": 30}))

    def test_removing_two_qubits(self):
        """Test removing two bits in an Histogram."""

        h = Histogram({"000": 40, "010": 30, "100": 20, "110": 10})
        h.remove_qubit_indices(0, 2)
        self.assertEqual(h, Histogram({"0": 60, "1": 40}))

    def test_post_select_method(self):
        """Post select in place based on one qubit measurement."""

        h = Histogram({"00": 40, "01": 30, "10": 20, "11": 10})
        h.post_select(expected_outcomes={0: "0"})
        self.assertEqual(h, Histogram({"0": 40, "1": 30}))

    def test_resample_method(self):
        """Test the resampling method. As resampling is probabilistic, we do not
        check for the probability outputs.
        """

        h = Histogram({"00": 60, "11": 40})

        h_10shots = h.resample(10)

        self.assertEqual(h_10shots.n_qubits, 2)
        self.assertEqual(h_10shots.n_shots, 10)

    def test_get_expectation_value_method(self):
        """Test the output for the get_expectation_value method."""

        h = Histogram({"00": 50, "11": 50})

        exp_1z = h.get_expectation_value(((0, "Z"),), 1.)
        self.assertAlmostEqual(exp_1z, 0.)

        exp_1zz = h.get_expectation_value(((0, "Z"), (1, "Z")), 1.)
        self.assertAlmostEqual(exp_1zz, 1.)

        exp_2zz = h.get_expectation_value(((0, "Z"), (1, "Z")), 2.)
        self.assertAlmostEqual(exp_2zz, 2.)


class AgregateHistogramsTest(unittest.TestCase):

    def test_aggregate_histograms(self):
        """Aggregate histograms into a new one."""

        h1 = Histogram({"00": 60, "11": 40})
        h2 = Histogram({"00": 60, "01": 40})

        hf = aggregate_histograms(h1, h2, h1)
        self.assertEqual(hf, Histogram({"00": 180, "11": 80, "01": 40}))

    def test_aggregate_histograms_length_inconsistent(self):
        """Return an error if histograms have bitstrings of different lengths."""
        with self.assertRaises(ValueError):
            h1 = Histogram({"00": 60, "11": 40})
            h2 = Histogram({"1": 100})
            aggregate_histograms(h1, h2)

    def test_aggregate_histograms_empty(self):
        """Return an error if no histogram is passed."""
        with self.assertRaises(ValueError):
            aggregate_histograms()


class FilterTest(unittest.TestCase):

    def test_filter_one_qubit(self):
        """Test filter Histogram based on one qubit measurement."""

        def f(bitstring, qubit_i, outcome):
            return bitstring[qubit_i] == outcome

        h = Histogram({"00": 40, "01": 30, "10": 20, "11": 10})

        h_ps_zero = filter_hist(h, f, 0, "0")
        self.assertEqual(h_ps_zero, Histogram({"00": 40, "01": 30}))

        h_ps_one = filter_hist(h, f, 0, "1")
        self.assertEqual(h_ps_one, Histogram({"10": 20, "11": 10}))

    def test_filter_many_qubits(self):
        """Test filter Histogram based on many qubit measurements."""

        def f(bitstring, qubit_i, outcome_i, qubit_j, outcome_j):
            return (bitstring[qubit_i] == outcome_i) and (bitstring[qubit_j] == outcome_j)

        h = Histogram({"000": 40, "010": 30, "101": 20, "111": 10})

        h_ps_zero = filter_hist(h, f, 0, "0", 1, "0")
        self.assertEqual(h_ps_zero, Histogram({"000": 40}))

        h_ps_one = filter_hist(h, f, 0, "1", 1, "1")
        self.assertEqual(h_ps_one, Histogram({"111": 10}))


if __name__ == "__main__":
    unittest.main()
