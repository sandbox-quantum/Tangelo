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

from tangelo.problem_decomposition import MIFNOHelper

pwd_this_test = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(pwd_this_test, "data")
mi_results = os.path.join(data_folder, "BeH2_STO3G_MIFNO_HBCI.json")

with open(mi_results, "r") as f:
    mi_object = json.loads(f.read())
mi_object["subproblem_data"] = {int(k): v for k, v in mi_object["subproblem_data"].items()}


class MIFNOHelperTest(unittest.TestCase):

    def test_init_from_file(self):
        """Verify initialization with a json file."""

        beh2_mifno = MIFNOHelper(mi_json_file=mi_results)

        self.assertAlmostEqual(beh2_mifno.e_tot, -15.595177739)
        self.assertAlmostEqual(beh2_mifno.e_corr, -0.034865396)
        self.assertAlmostEqual(beh2_mifno.e_mf, -15.560312343)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 2)

    def test_init_from_dict(self):
        """Verify initialization with a dict object."""

        beh2_mifno = MIFNOHelper(mi_dict=mi_object)

        self.assertAlmostEqual(beh2_mifno.e_tot, -15.595177739)
        self.assertAlmostEqual(beh2_mifno.e_corr, -0.034865396)
        self.assertAlmostEqual(beh2_mifno.e_mf, -15.560312343)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_mifno.frag_info), 2)

    def test_fragment_ids(self):
        """Verify if the fragment_ids property returns all the fragment ids.."""

        beh2_mifno = MIFNOHelper(mi_dict=mi_object)
        frag_ids = beh2_mifno.fragment_ids

        self.assertEqual(frag_ids, ["(0,)", "(1,)", "(2,)", "(0, 1)", "(0, 2)", "(1, 2)"])

    def test_mi_summation(self):
        """Verify that the energy can be recomputed with the incremental method."""

        beh2_mifno = MIFNOHelper(mi_dict=mi_object)

        e_mi = beh2_mifno.mi_summation()

        self.assertAlmostEqual(e_mi, beh2_mifno.e_tot)


if __name__ == "__main__":
    unittest.main()
