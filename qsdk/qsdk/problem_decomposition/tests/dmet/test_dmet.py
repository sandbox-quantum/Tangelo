import unittest
from pyscf import gto

from qsdk.problem_decomposition.dmet.dmet_problem_decomposition import Localization, DMETProblemDecomposition

H10_RING = """
        H      0.970820393250   0.000000000000   0.000000000000
        H      0.785410196625   0.570633909777   0.000000000000
        H      0.300000000000   0.923305061153   0.000000000000
        H     -0.300000000000   0.923305061153   0.000000000000
        H     -0.785410196625   0.570633909777   0.000000000000
        H     -0.970820393250   0.000000000000   0.000000000000
        H     -0.785410196625  -0.570633909777   0.000000000000
        H     -0.300000000000  -0.923305061153   0.000000000000
        H      0.300000000000  -0.923305061153   0.000000000000
        H      0.785410196625  -0.570633909777   0.000000000000
        """

H4_RING = """
        H   0.7071067811865476   0.0                 0.0
        H   0.0                  0.7071067811865476  0.0
        H  -1.0071067811865476   0.0                 0.0
        H   0.0                 -1.0071067811865476  0.0
        """


class DMETProblemDecompositionTest(unittest.TestCase):

    def test_incorrect_number_atoms(self):
        """Tests if the program raises the error when the number of
        fragment sites is not equal to the number of atoms in the molecule."""
        mol = gto.Mole()
        mol.atom = H10_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        # The molecule has more atoms than this.
        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)

    def test_incorrect_number_solvers(self):
        """Tests if the program raises the error when the number of
        fragment sites is not equal to the number of solvers."""
        mol = gto.Mole()
        mol.atom = H10_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [2, 3, 2, 3],
                    "fragment_solvers": ["fci", "fci"],
                    "verbose": False
                    }

        with self.assertRaises(RuntimeError):
            DMETProblemDecomposition(opt_dmet)

    def test_h10ring_ml_fci_no_mf(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and FCI solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H10_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1]*10,
                    "fragment_solvers": "fci",
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -4.498973024, places=4)

    def test_h4ring_ml_ccsd_no_mf_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and CCSD solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
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
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and default solver (currently CCSD) for fragments """
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()
        energy = dmet_solver.simulate()

        self.assertAlmostEqual(energy, -0.854379, places=6)

    def test_h4ring_ml_fci_no_mf_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and FCI solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
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
        chemical accuracy."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "3-21g"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": ['fci', 'fci', 'ccsd', 'ccsd'],
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        solver = DMETProblemDecomposition(opt_dmet)
        solver.build()
        energy = solver.simulate()
        self.assertAlmostEqual(energy, -0.94199, places=4)

    def test_fragment_ids(self):
        """Tests if a nested list of atom ids is provided. """
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "3-21g"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [[0], [1], [2], [3]],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        solver = DMETProblemDecomposition(opt_dmet)

        self.assertEqual(solver.fragment_atoms, [1, 1, 1, 1])

    def test_fragment_ids_exceptions(self):
        """Tests exceptions if a bad nested list of atom ids is provided.
        2 cases: if an atom id is higher than the number of atoms and if an
        id is detected twice (or more).
        """
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "3-21g"
        mol.charge = 2
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [[0,0], [1], [2], [3]],
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
