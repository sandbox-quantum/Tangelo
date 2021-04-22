import unittest
from pyscf import gto

from qsdk.problem_decomposition import DMETProblemDecomposition
from qsdk.problem_decomposition.electron_localization import iao_localization, meta_lowdin_localization

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

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = meta_lowdin_localization
        # The molecule has more atoms than this.
        self.assertRaises(RuntimeError, dmet_solver.simulate, mol, [1, 1, 1, 1])

    def test_incorrect_number_solvers(self):
        """Tests if the program raises the error when the number of
        fragment sites is not equal to the number of solvers."""
        mol = gto.Mole()
        mol.atom = H10_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        dmet_solver = DMETProblemDecomposition()
        self.assertRaises(RuntimeError, dmet_solver.simulate, mol, [2, 3, 2, 3], fragment_solvers=['fci'])

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

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = meta_lowdin_localization
        energy = dmet_solver.simulate(mol, [1]*10, fragment_solvers=['fci']*10)

        self.assertAlmostEqual(energy, -4.498973024, places=4)

    def test_h4ring_ml_ccsd_no_mf_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and CCSD solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = meta_lowdin_localization
        energy = dmet_solver.simulate(mol, [1, 1, 1, 1], fragment_solvers=['ccsd']*4)

        self.assertAlmostEqual(energy, -1.9916120594, places=6)

    def test_h4ring_ml_default_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and default solver (currently CCSD) for fragments """
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = meta_lowdin_localization
        energy = dmet_solver.simulate(mol, [1, 1, 1, 1])

        self.assertAlmostEqual(energy, -1.9916120594, places=6)

    def test_h4ring_ml_fci_no_mf_minao(self):
        """ Tests the result from DMET against a value from a reference
        implementation with meta-lowdin localization and FCI solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = meta_lowdin_localization
        energy = dmet_solver.simulate(mol, [1, 1, 1, 1], fragment_solvers=['fci']*4)

        self.assertAlmostEqual(energy, -1.9916120594, places=4)

    def test_solver_mix(self):
        """Tests that solving with multiple solvers works.

        With this simple system, we can assume that both CCSD and FCI can reach
        chemical accuracy."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        solver = DMETProblemDecomposition()
        solver.electron_localization_method = iao_localization
        energy = solver.simulate(mol, [1, 1, 1, 1], fragment_solvers=['fci', 'fci', 'ccsd', 'ccsd'])
        self.assertAlmostEqual(energy, -2.0284, places=4)

    @unittest.skip("Behavior currently non deterministic. Newton solver in DMET sometimes fails to converge "
                   "Changing tolerance or initial guess may help")
    def test_h4ring_iao_ccsd_no_mf_321g(self):
        """ Tests the result from DMET against a value from a reference
        implementation with IAO localization, 3-21g basis, and CCSD solution to
        fragments."""
        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        dmet_solver = DMETProblemDecomposition()
        dmet_solver.electron_localization_method = iao_localization
        energy = dmet_solver.simulate(mol, [2, 2], fragment_solvers=['ccsd']*2)

        self.assertAlmostEqual(energy, -2.0290205366, places=6)


if __name__ == "__main__":
    unittest.main()
