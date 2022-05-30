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

from tangelo.toolboxes.molecular_computation.mifno_link import MIFNOFragment

pwd_this_test = os.path.dirname(os.path.abspath(__file__))

# Dictionary containing all the results.
with open(os.path.join(pwd_this_test, "data", "BeH2_STO3G_3MIFNO_FCI.json"), "r") as f:
    res = json.loads(f.read())

# json export converted some int keys to string (not wanted).
res["subproblem_data"] = {int(k): v for k, v in res["subproblem_data"].items()}


class MIFNOFragmentTest(unittest.TestCase):

    def test_init(self):
        """Verify initialization."""

        test = MIFNOFragment(res)

        self.assertAlmostEquals(test.e_tot, -15.595176868)
        self.assertAlmostEquals(test.e_corr, -0.034864526)
        self.assertAlmostEquals(test.e_mf, -15.560312342)

        # Testing the number of detected increments.
        self.assertEqual(len(test.frag_info), 3)

        # Testing the total number of fragments.
        self.assertEqual(len(test.frag_info_flattened), 7)

    def test_fragment_ids(self):
        """Verify if the fragment_ids property returns all the fragment ids.."""

        test = MIFNOFragment(res)
        frag_ids = test.fragment_ids

        self.assertEquals(frag_ids, ["(0,)", "(1,)", "(2,)", "(0, 1)", "(0, 2)", "(1, 2)", "(0, 1, 2)"])

    def test_mi_summation(self):
        """Verify that the energy can be recomputed with the incremental method."""

        test = MIFNOFragment(res)
        test.retrieve_mo_coeff(os.path.join(pwd_this_test, "data"))

        e_mi = test.mi_summation()

        self.assertAlmostEqual(e_mi, test.e_tot)


if __name__ == "__main__":
    unittest.main()
