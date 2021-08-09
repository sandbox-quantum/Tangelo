import unittest

from qsdk.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from qsdk.problem_decomposition.oniom._helpers.helper_classes import Fragment, Link

H4 = [('H', (0., 0., 0.)), ('H', (0., 0., 0.75)),
      ('H', (0., 0., 2.)), ('H', (0., 0., 2.75))]

PHE = """
N      0.7060     -1.9967     -0.0757
C      1.1211     -0.6335     -0.4814
C      0.6291      0.4897      0.4485
C     -0.8603      0.6071      0.4224
C     -1.4999      1.1390     -0.6995
C     -2.8840      1.2600     -0.7219
C     -3.6384      0.8545      0.3747
C     -3.0052      0.3278      1.4949
C     -1.6202      0.2033      1.5209
C      2.6429     -0.5911     -0.5338
O      3.1604     -0.2029     -1.7213
O      3.4477     -0.8409      0.3447
H     -0.2916     -2.0354     -0.0544
H      1.0653     -2.2124      0.8310
H      0.6990     -0.4698     -1.5067
H      1.0737      1.4535      0.1289
H      0.9896      0.3214      1.4846
H     -0.9058      1.4624     -1.5623
H     -3.3807      1.6765     -1.6044
H     -4.7288      0.9516      0.3559
H     -3.5968      0.0108      2.3601
H     -1.1260     -0.2065      2.4095
H      4.1118     -0.2131     -1.6830
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

        with self.assertRaises(NotImplementedError):
            oniom_solver = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model]})

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
        geom_fragment_capped = oniom_solver.fragments[1].geometry

        # Those position have been verified with a molecular visualization software.
        PHE_backbone_capped = [('N', (0.706, -1.9967, -0.0757)),
                               ('C', (1.1211, -0.6335, -0.4814)),
                               ('C', (2.6429, -0.5911, -0.5338)),
                               ('O', (3.1604, -0.2029, -1.7213)),
                               ('O', (3.4477, -0.8409, 0.3447)),
                               ('H', (-0.2916, -2.0354, -0.0544)),
                               ('H', (1.0653, -2.2124, 0.831)),
                               ('H', (0.699, -0.4698, -1.5067)),
                               ('H', (4.1118, -0.2131, -1.683)),
                               ('H', (0.772272, 0.1628488, 0.1778991))]

        # Every atom must be the same (same order too).
        # Position can be almost equals.
        for i, atom in enumerate(geom_fragment_capped):
            self.assertEquals(atom[0], PHE_backbone_capped[i][0])

            for dim in range(3):
                self.assertAlmostEquals(atom[1][dim], PHE_backbone_capped[i][1][dim])

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

        self.assertAlmostEquals(e_oniom, -544.7306189, places=6)

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
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        # Comparing VQE-UCCSD to CCSD.
        system = Fragment(solver_low="RHF", options_low=options_both)
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
        oniom_model_cc = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model_cc_1, model_cc_2]})

        e_oniom_vqe = oniom_model_vqe.simulate()
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

        self.assertAlmostEquals(e_oniom, -315.23418566258914, places=6)

    def test_get_resources(self):
        """Test to verifiy the implementation of resources estimation in ONIOM. """

        options_both = {"basis": "sto-3g"}

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
        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": H4, "fragments": [system, model_vqe_1, model_vqe_2]})

        vqe_resources = {"qubit_hamiltonian_terms": 15,
                         "circuit_width": 4,
                         "circuit_gates": 158,
                         "circuit_2qubit_gates": 64,
                         "circuit_var_gates": 12,
                         "vqe_variational_parameters": 2}

        res = oniom_model_vqe.get_resources()

        self.assertEqual(res[1], vqe_resources)
        self.assertEqual(res[2], vqe_resources)


if __name__ == "__main__":
    unittest.main()
