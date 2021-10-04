import unittest

from qsdk.molecule_library import mol_H4_doublecation_minao, mol_H4_doublecation_321g, mol_H10_321g
from qsdk.problem_decomposition.dmet.dmet_problem_decomposition import Localization, DMETProblemDecomposition


class DMETProblemDecompositionTest(unittest.TestCase):

    def test_incorrect_number_atoms(self):
        """Tests if the program raises the error when the number of fragment
        sites is not equal to the number of atoms in the molecule.
        """

        opt_dmet = {"molecule": mol_H10_321g,
                    "fragment_atoms": [1, 1, 1, 1],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        # The molecule has more atoms than this.
        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)

    def test_incorrect_number_solvers(self):
        """Tests if the program raises the error when the number of fragment
        sites is not equal to the number of solvers.
        """

        opt_dmet = {"molecule": mol_H10_321g,
                    "fragment_atoms": [2, 3, 2, 3],
                    "fragment_solvers": ["fci", "fci"],
                    "verbose": False
                    }

        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)

    def test_h10ring_ml_fci(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and FCI solution to
        fragments."""

        opt_dmet = {"molecule": mol_H10_321g,
                    "fragment_atoms": [1]*10,
                    "fragment_solvers": "fci",
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -4.498973024, places=4)

    def test_h4ring_ml_ccsd_minao(self):
        """Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and CCSD solution to
        fragments.
        """

        opt_dmet = {"molecule": mol_H4_doublecation_minao,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -0.854379, places=6)

    def test_h4ring_ml_default_minao(self):
        """Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and default solver
        (currently CCSD) for fragments.
        """

        opt_dmet = {"molecule": mol_H4_doublecation_minao,
                    "fragment_atoms": [1, 1, 1, 1],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -0.854379, places=6)

    def test_h4ring_ml_fci_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and FCI solution to
        fragments.
        """

        opt_dmet = {"molecule": mol_H4_doublecation_minao,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": "fci",
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -0.854379, places=4)

    def test_solver_mix(self):
        """Tests that solving with multiple solvers works.

        With this simple system, we can assume that both CCSD and FCI can reach
        chemical accuracy.
        """

        opt_dmet = {"molecule": mol_H4_doublecation_321g,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": ["fci", "fci", "ccsd", "ccsd"],
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        solver = DMETProblemDecomposition(opt_dmet)
        solver.build()
        energy = solver.simulate()
        self.assertAlmostEqual(energy, -0.94199, places=4)

    def test_fragment_ids(self):
        """Tests if a nested list of atom ids is provided."""

        opt_dmet = {"molecule": mol_H4_doublecation_321g,
                    "fragment_atoms": [[0], [1], [2], [3]],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        solver = DMETProblemDecomposition(opt_dmet)

        self.assertEqual(solver.fragment_atoms, [1, 1, 1, 1])

    def test_fragment_ids_exceptions(self):
        """Tests exceptions if a bad nested list of atom ids is provided. Two
        cases: if an atom id is higher than the number of atoms and if an id is
        detected twice (or more).
        """

        opt_dmet = {"molecule": mol_H4_doublecation_321g,
                    "fragment_atoms": [[0, 0], [1], [2], [3]],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)

        opt_dmet["fragment_atoms"] = [[0], [1], [2], [4]]

        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)


if __name__ == "__main__":
    unittest.main()
