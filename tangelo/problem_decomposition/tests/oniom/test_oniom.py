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

from numpy import linspace

from tangelo.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment, Link
from tangelo.molecule_library import xyz_H4, xyz_PHE


class ONIOMTest(unittest.TestCase):

    def test_selected_atom(self):
        """Test selection of atoms. If it is not a list of int or an int, an
        error should be raised.
        """

        with self.assertRaises(TypeError):
            ONIOMProblemDecomposition({"geometry": xyz_H4,
                                       "fragments": [Fragment(solver_low="HF", solver_high="CCSD", selected_atoms=[3.1415])]})

    def test_not_implemented_solver(self):
        # Definition of simple fragments to test the error raising.

        with self.assertRaises(NotImplementedError):
            Fragment(solver_low="UNSUPPORTED")

        with self.assertRaises(NotImplementedError):
            Fragment(solver_low="HF",
                     solver_high="UNSUPPORTED",
                     selected_atoms=[0, 1])

    def test_capping_broken_link(self):
        """Testing the positon of a new H atom when a bond is broken."""

        system = Fragment(solver_low="HF")

        link = [Link(1, 2, 0.709, "H")]
        model = Fragment(solver_low="HF",
                         solver_high="CCSD",
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": xyz_PHE, "fragments": [system, model]})

        # Retrieving fragment geometry with an H atom replacing a broken bond.
        geom_fragment_capped = oniom_solver.fragments[1].geometry

        # Those position have been verified with a molecular visualization software.
        PHE_backbone_capped = [("N", (0.706, -1.9967, -0.0757)),
                               ("C", (1.1211, -0.6335, -0.4814)),
                               ("C", (2.6429, -0.5911, -0.5338)),
                               ("O", (3.1604, -0.2029, -1.7213)),
                               ("O", (3.4477, -0.8409, 0.3447)),
                               ("H", (-0.2916, -2.0354, -0.0544)),
                               ("H", (1.0653, -2.2124, 0.831)),
                               ("H", (0.699, -0.4698, -1.5067)),
                               ("H", (4.1118, -0.2131, -1.683)),
                               ("H", (0.772272, 0.1628488, 0.1778991))]

        # Every atom must be the same (same order too).
        # Position can be almost equals.
        for i, atom in enumerate(geom_fragment_capped):
            self.assertEqual(atom[0], PHE_backbone_capped[i][0])

            for dim in range(3):
                self.assertAlmostEqual(atom[1][dim], PHE_backbone_capped[i][1][dim], places=4)

    def test_energy_hf_ccsd_h4(self):
        """Testing the oniom energy with a low accuracy method (HF) and an
        higher one (CCSD) for H4 molecule. The H2-H2 interaction is computed at
        the HF level.
        """

        options_both = {"basis": "sto-3g"}

        system = Fragment(solver_low="HF", options_low=options_both)
        model_cc_1 = Fragment(solver_low="HF",
                              options_low=options_both,
                              solver_high="CCSD",
                              options_high=options_both,
                              selected_atoms=[0, 1])
        model_cc_2 = Fragment(solver_low="HF",
                              options_low=options_both,
                              solver_high="CCSD",
                              options_high=options_both,
                              selected_atoms=[2, 3])
        oniom_model_cc = ONIOMProblemDecomposition({"geometry": xyz_H4, "fragments": [system, model_cc_1, model_cc_2]})

        e_oniom_cc = oniom_model_cc.simulate()
        self.assertAlmostEqual(-1.901616, e_oniom_cc, places=5)

    def test_energy_hf_ccsd_phe(self):
        """Testing the oniom energy with a low accuracy method (HF) and an
        higher one (CCSD) for PHE molecule. The important fragment is chosen to
        be the backbone. The side chain is computed at the HF level.
        """

        options_low = {"basis": "sto-3g"}
        options_high = {"basis": "sto-3g"}

        system = Fragment(solver_low="HF", options_low=options_low)

        link = [Link(1, 2, 0.709, "H")]
        model = Fragment(solver_low="HF",
                         options_low=options_low,
                         solver_high="CCSD",
                         options_high=options_high,
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": xyz_PHE, "fragments": [system, model]})
        e_oniom = oniom_solver.simulate()

        self.assertAlmostEqual(e_oniom, -544.730619, places=4)

    def test_energy_multilayers(self):
        """Testing the oniom energy with a low accuracy method (HF), a medium
        accuracy (CCSD) and an higher one (FCI) for a H9 chain.
        """

        # H9 chain.
        xyz_h9 = [("H", (x, 0., 0.)) for x in linspace(-2., 2., num=9)]

        options = {"basis": "sto-3g"}

        # All system in HF.
        system = Fragment(solver_low="HF", options_low=options, spin=1)

        # Central 3 H energy is computed with FCI.
        high = Fragment(solver_low="CCSD",
                        options_low=options,
                        solver_high="FCI",
                        options_high=options,
                        selected_atoms=[3, 4, 5],
                        spin=1)

        # 2 H "buffer" atoms energy is computed with CCSD.
        medium = Fragment(solver_low="HF",
                          options_low=options,
                          solver_high="CCSD",
                          options_high=options,
                          selected_atoms=[2, 3, 4, 5, 6],
                          spin=1)

        oniom_solver = ONIOMProblemDecomposition({"geometry": xyz_h9, "fragments": [system, medium, high]})
        e_oniom = oniom_solver.simulate()

        self.assertAlmostEqual(-2.925695, e_oniom, places=5)


if __name__ == "__main__":
    unittest.main()
