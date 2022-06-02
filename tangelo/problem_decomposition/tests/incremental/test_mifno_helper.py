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
import os
import json

from tangelo.problem_decomposition import MIFNOHelper

pwd_this_test = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(pwd_this_test, "data", "BeH2_STO3G_3MIFNO_FCI.json")

with open(json_file, "r") as f:
    results_object = json.loads(f.read())
results_object["subproblem_data"] = {int(k): v for k, v in results_object["subproblem_data"].items()}


class MIFNOHelperTest(unittest.TestCase):

    def test_init_from_file(self):
        """Verify initialization with a json file."""

        beh2_mifno = MIFNOHelper(json_file)

        self.assertAlmostEquals(beh2_mifno.e_tot, -15.595176868)
        self.assertAlmostEquals(beh2_mifno.e_corr, -0.034864526)
        self.assertAlmostEquals(beh2_mifno.e_mf, -15.560312342)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 3)

    def test_init_from_dict(self):
        """Verify initialization with a dict object."""

        beh2_mifno = MIFNOHelper(results_object=results_object)

        self.assertAlmostEquals(beh2_mifno.e_tot, -15.595176868)
        self.assertAlmostEquals(beh2_mifno.e_corr, -0.034864526)
        self.assertAlmostEquals(beh2_mifno.e_mf, -15.560312342)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 3)

    def test_fragment_ids(self):
        """Verify if the fragment_ids property returns all the fragment ids.."""

        beh2_mifno = MIFNOHelper(json_file)
        frag_ids = beh2_mifno.fragment_ids

        self.assertEquals(frag_ids, ["(0,)", "(1,)", "(2,)", "(0, 1)", "(0, 2)", "(1, 2)", "(0, 1, 2)"])

    def test_mi_summation(self):
        """Verify that the energy can be recomputed with the incremental method."""

        beh2_mifno = MIFNOHelper(json_file)
        beh2_mifno.retrieve_mo_coeff(os.path.join(pwd_this_test, "data"))

        e_mi = beh2_mifno.mi_summation()

        self.assertAlmostEqual(e_mi, beh2_mifno.e_tot)


if __name__ == "__main__":
    unittest.main()
