import unittest

from qsdk.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from qsdk.problem_decomposition.oniom._helpers.helper_classes import Fragment, Link

H4 = [('H', (0., 0., 0.)), ('H', (0., 0., 0.75)),
      ('H', (0., 0., 2.)), ('H', (0., 0., 2.75))]

PHE = """
N     0.68274   -2.04442   -0.00262
C     1.12932   -0.71878   -0.50109
C     0.65219    0.50498    0.34793
C    -0.87144    0.63909    0.37265
C    -1.54896    1.17847   -0.71891
C    -2.92863    1.28804   -0.71077
C    -3.65534    0.85827    0.39099
C    -2.99217    0.32145    1.48162
C    -1.60909    0.21252    1.47096
C     2.67929   -0.72562   -0.55612
O     3.16186    0.14811   -1.52435
O     3.41930   -1.37436    0.15808
H    -0.33407   -1.98612    0.16691
H     1.10677   -2.18182    0.92918
H     0.76187   -0.59639   -1.52257
H     1.09357    1.40375   -0.07905
H     1.02860    0.39932    1.36347
H    -0.98834    1.51875   -1.58040
H    -3.44085    1.71133   -1.56567
H    -4.73426    0.94475    0.39800
H    -3.55144   -0.01364    2.34607
H    -1.09895   -0.20287    2.33068
H     4.14692    0.07049   -1.46330
"""


class ONIOMTest(unittest.TestCase):

    def test_selected_atom(self):
        """Test selection of atoms. If it is not a list of int or an int, an error
        should be raised.
        """

        # Definition of simple fragments to test the error raising.
        system = Fragment(solver_low="RHF")
        model = Fragment(solver_low="RHF",
                         solver_high="RHF",
                         # Next line should be problematic (float number).
                         selected_atoms=[3.1415])

        with self.assertRaises(TypeError):
            ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model]})

    def test_not_implemented_solver(self):
        # Definition of simple fragments to test the error raising.
        system = Fragment(solver_low="RHF")
        model = Fragment(solver_low="RHF",
                         solver_high="BANANA",
                         selected_atoms=[0, 1])

        ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model]})

        with self.assertRaises(NotImplementedError):
            oniom_solver = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model]})
            oniom_solver.simulate()

    def test_capping_broken_link(self):
        """Testing the positon of a new H atom when a bond is broken. """

        system = Fragment(solver_low="RHF")

        link = [Link(1, 2, 0.709, 'H')]
        model = Fragment(solver_low="RHF",
                         solver_high="CCSD",
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": PHE, "fragments": [system, model]})

        # Retrieving fragment geometry with an H atom replacing a broken bond.
        # Only getting the addition H.
        hydrogen_cap = [oniom_solver.fragments[1].geometry[-1]]

        # Those position have been verified with a molecular visualization software.
        hydrogen_cap_ref = [('H', (1.494839, 0.281315, 0.190607))]

        # Internally, coordinates are in bohrs.
        #hydrogen_cap_ref = oniom_solver.angstrom_to_bohr(hydrogen_cap_ref)

        # Position can be almost equals.
        for dim in range(3):
            self.assertAlmostEquals(hydrogen_cap_ref[0][1][dim], hydrogen_cap[0][1][dim], places=5)

    def test_energy(self):
        """Testing the oniom energy with a low accuraccy method (RHF) and an
        higher one (CCSD) for PHE molecule. The important fragment is
        chosen to be the backbone. The side chain is computed at the RHF
        level.
        """

        options_low = {"basis": "sto-3g"}
        options_high = {"basis": "sto-3g"}

        system = Fragment(solver_low="RHF", options_low=options_low)

        link = [Link(1, 2, 0.709, 'H')]
        model = Fragment(solver_low="RHF",
                         options_low=options_low,
                         solver_high="CCSD",
                         options_high=options_high,
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": PHE, "fragments": [system, model]})
        e_oniom = oniom_solver.simulate()

        self.assertAlmostEquals(e_oniom, -544.7577848720764, places=6)

    def test_vqe_cc(self):
        """Test to verifiy the implementation of VQE (with UCCSD) in ONIOM. Results
        between VQE-UCCSD and CCSD should be the same.
        """

        options_both = {"basis": "sto-3g"}

        # With this line, the interaction between H2-H2 is computed with a low
        # accuracy method.
        system = Fragment(solver_low="RHF", options_low=options_both)

        # VQE-UCCSD fragments.
        model_vqe_1 = Fragment(solver_low="RHF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
                               selected_atoms=[0, 1])
        model_vqe_2 = Fragment(solver_low="RHF",
                               options_low=options_both,
                               solver_high="VQE",
                               options_high=options_both,
                               selected_atoms=[2, 3])

        # Comparing VQE-UCCSD to CCSD.
        model_cc_1 = Fragment(solver_low="RHF",
                              options_low=options_both,
                              solver_high="CCSD",
                              options_high=options_both,
                              selected_atoms=[0, 1])
        model_cc_2 = Fragment(solver_low="RHF",
                              options_low=options_both,
                              solver_high="CCSD",
                              options_high=options_both,
                              selected_atoms=[2, 3])

        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model_vqe_1, model_vqe_2]})
        e_oniom_vqe = oniom_model_vqe.simulate()

        oniom_model_cc = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model_cc_1, model_cc_2]})
        e_oniom_cc = oniom_model_cc.simulate()

        # The two results (VQE-UCCSD and CCSD) should be more or less the same.
        self.assertAlmostEqual(e_oniom_vqe, e_oniom_cc, places=5)

    def test_semi_empirical_mindo3_link(self):
        """Testing the oniom link with the semi-empirical electronic solver
        MINDO3.
        """

        # For semi-empirical solver, no basis set is required.
        # There is no options_low in this context.
        options_high = {"basis": "sto-3g"}

        system = Fragment(solver_low="MINDO3")

        link = [Link(1, 2, 0.709, 'H')]
        model = Fragment(solver_low="MINDO3",
                         solver_high="CCSD",
                         options_high=options_high,
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": PHE, "fragments": [system, model]})
        e_oniom = oniom_solver.simulate()

        self.assertAlmostEquals(e_oniom, -315.2538461038537, places=5)

    def test_geom_optimization(self):
        """Testing the oniom geometry optimization capability. """

        options_low = {"basis": "sto-3g"}
        options_high = {"basis": "sto-3g"}

        system = Fragment(solver_low="RHF", options_low=options_low)

        link = [Link(1, 2, 0.709, 'H')]
        model = Fragment(solver_low="RHF",
                         options_low=options_low,
                         solver_high="RHF",
                         options_high=options_high,
                         selected_atoms=[0, 1, 9, 10, 11, 12, 13, 14, 22],
                         broken_links=link)

        oniom_solver = ONIOMProblemDecomposition({"geometry": PHE, "fragments": [system, model]})
        oniom_solver.optimize()


if __name__ == "__main__":
    unittest.main()
