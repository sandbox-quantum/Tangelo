import unittest

from qsdk.problem_decomposition.oniom.oniom_problem_decomposition import ONIOMProblemDecomposition
from qsdk.problem_decomposition.oniom._helpers.helper_classes import Fragment


class ONIOMTest(unittest.TestCase):

    def test_vqe_cc(self):
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.75)),
                 ('H', (0., 0., 2.)), ('H', (0., 0., 2.75))]

        options_both = {"basis": "sto-3g"}

        # With this line, the interaction between H2-H2 is computed with a low
        # method.
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

        # Compairing VQE-UCCSD to CCSD.
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

        oniom_model_vqe = ONIOMProblemDecomposition({"geometry": geometry, "fragments": [system, model_vqe_1, model_vqe_2]})
        e_oniom_vqe = oniom_model_vqe.simulate()

        oniom_model_cc = ONIOMProblemDecomposition({"geometry": geometry, "fragments": [system, model_cc_1, model_cc_2]})
        e_oniom_cc = oniom_model_cc.simulate()

        # The two results (VQE-UCCSD and CCSD) should be more or less the same.
        self.assertAlmostEqual(e_oniom_vqe, e_oniom_cc, places=5)


if __name__ == "__main__":
    unittest.main()
