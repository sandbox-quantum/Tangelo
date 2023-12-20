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
mi_results = os.path.join(data_folder, "BeH2_CCPVDZ_iFCI_HBCI/full_results_13242875561568846089.log")

with open(mi_results, "r") as f:
    mi_object = json.loads("\n".join(f.readlines()[1:]))
mi_object["subproblem_data"] = {int(k): v for k, v in mi_object["subproblem_data"].items()}


class IFCIHelperTest(unittest.TestCase):

    def setUp(self):
        self.e_tot = -15.834586605234621
        self.e_corr = -0.06745129202803568
        self.e_mf = self.e_tot - self.e_corr

    def test_init_from_file(self):
        """Verify initialization with a json file."""

        beh2_ifci = MethodOfIncrementsHelper(log_file=mi_results)

        self.assertAlmostEqual(beh2_ifci.e_tot, self.e_tot)
        self.assertAlmostEqual(beh2_ifci.e_corr, self.e_corr)
        self.assertAlmostEqual(beh2_ifci.e_mf, self.e_mf)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_ifci.frag_info), 2)

    def test_init_from_dict(self):
        """Verify initialization with a dict object."""

        beh2_ifci = MethodOfIncrementsHelper(full_result=mi_object)

        self.assertAlmostEqual(beh2_ifci.e_tot, self.e_tot)
        self.assertAlmostEqual(beh2_ifci.e_corr, self.e_corr)
        self.assertAlmostEqual(beh2_ifci.e_mf, self.e_mf)

        # Testing the number of detected increments.
        self.assertEqual(len(beh2_ifci.frag_info), 2)

    def test_fragment_ids(self):
        """Verify whether fragment_ids property returns all the fragment ids."""

        beh2_ifci = MethodOfIncrementsHelper(full_result=mi_object)
        frag_ids = beh2_ifci.fragment_ids

        self.assertEqual(frag_ids, ["(0,)", "(1,)", "(2,)", "(0, 1)", "(0, 2)", "(1, 2)"])

    def test_mi_summation(self):
        """Verify that the energy can be recomputed with the incremental method."""

        beh2_ifci = MethodOfIncrementsHelper(full_result=mi_object)
        e_mi = beh2_ifci.mi_summation()

        self.assertAlmostEqual(e_mi, beh2_ifci.e_tot)

    def test_retrieve_mo_coeff(self):
        """Verify that the molecular orbital coefficients can be read from files."""

        beh2_ifci = MethodOfIncrementsHelper(full_result=mi_object)
        beh2_ifci.retrieve_mo_coeff(source_folder=os.path.join(data_folder, "BeH2_CCPVDZ_iFCI_HBCI"),
                                    prefix="mo_coefficients_")

    def test_missing_folder(self):
        """Verify that a missing folder for molecular orbital coefficients is
        raising an error.
        """

        beh2_ifci = MethodOfIncrementsHelper(full_result=mi_object)

        with self.assertRaises(FileNotFoundError):
            beh2_ifci.retrieve_mo_coeff(source_folder=os.path.join(data_folder, "missing_folder"),
                                        prefix="mo_coefficients_")


if __name__ == "__main__":
    unittest.main()
