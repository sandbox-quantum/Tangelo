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
import os
import json

from tangelo.problem_decomposition import MethodOfIncrementsHelper

pwd_this_test = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(pwd_this_test, "data")
mi_results = os.path.join(data_folder, "BeH2_CCPVDZ_MIFNO_HBCI/full_results_186184445859680125.log")

with open(mi_results, "r") as f:
    mi_object = json.loads("\n".join(f.readlines()[1:]))
mi_object["subproblem_data"] = {int(k): v for k, v in mi_object["subproblem_data"].items()}


class MIFNOHelperTest(unittest.TestCase):

    def setUp(self):
        self.e_tot = -15.83647358459995
        self.e_corr = -0.06915435696141081
        self.e_mf = self.e_tot - self.e_corr

    def test_init_from_file(self):
        """Verify initialization with a json file."""

        beh2_mifno = MethodOfIncrementsHelper(log_file=mi_results)

        self.assertAlmostEqual(beh2_mifno.e_tot, self.e_tot)
        self.assertAlmostEqual(beh2_mifno.e_corr, self.e_corr)
        self.assertAlmostEqual(beh2_mifno.e_mf, self.e_mf)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 2)

    def test_init_from_dict(self):
        """Verify initialization with a dict object."""

        beh2_mifno = MethodOfIncrementsHelper(full_result=mi_object)

        self.assertAlmostEqual(beh2_mifno.e_tot, self.e_tot)
        self.assertAlmostEqual(beh2_mifno.e_corr, self.e_corr)
        self.assertAlmostEqual(beh2_mifno.e_mf, self.e_mf)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 2)

    def test_fragment_ids(self):
        """Verify whether fragment_ids property returns all the fragment ids."""

        beh2_mifno = MethodOfIncrementsHelper(full_result=mi_object)
        frag_ids = beh2_mifno.fragment_ids

        self.assertEqual(frag_ids, ["(0,)", "(1,)", "(2,)", "(0, 1)", "(0, 2)", "(1, 2)"])

    def test_mi_summation(self):
        """Verify that the energy can be recomputed with the incremental method."""

        beh2_mifno = MethodOfIncrementsHelper(full_result=mi_object)

        e_mi = beh2_mifno.mi_summation()

        self.assertAlmostEqual(e_mi, beh2_mifno.e_tot)


if __name__ == "__main__":
    unittest.main()
