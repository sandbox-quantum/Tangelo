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

import unittest

from tangelo.toolboxes.post_processing import Histogram, aggregate_histograms

d1 = {'00': 0.6, '11': 0.4}
d2 = {'00': 0.2}
d3 = {'00': 0.6, '111': 0.4}
d4 = {'00': 0.6, '01': 0.4}
d5 = {'0': 1.}


class HistogramTest(unittest.TestCase):

    def test_init_histogram(self):
        """ Initialize histogram object
        """

        n_shots = 100
        h1 = Histogram(d1, n_shots)
        self.assertTrue(h1.freqs == d1)
        self.assertTrue(h1.n_shots == n_shots)

    def test_histogram_inconsistent_bitstring_length(self):
        """ Throw an error if bitstrings are not all of consistent length
        """
        with self.assertRaises(ValueError):
            h = Histogram(d3, 1)

    def test_aggregate_histograms(self):
        """ Aggregate histograms into a new one
        """
        h1 = Histogram(d1, 100)
        h2 = Histogram(d4, 200)

        hf = aggregate_histograms(h1, h2, h1)
        self.assertTrue(hf == Histogram({'00': 0.6, '11': 0.2, '01': 0.2}, 400))

    def test_aggregate_histograms_length_inconsistent(self):
        """ Return an error if histograms have bitstrings of different lengths
        """
        with self.assertRaises(ValueError):
            h1 = Histogram(d4, 1)
            h2 = Histogram(d5, 1)
            hf = aggregate_histograms(h1, h2)

    def test_aggregate_histograms_empty(self):
        """ Return an error if no histogram is passed
        """
        with self.assertRaises(ValueError):
            hf = aggregate_histograms()


if __name__ == "__main__":
    unittest.main()
