import unittest

from pyscf import gto

from qsdk.electronic_structure_solvers import FCISolver, VQESolver
from qsdk.electronic_structure_solvers.vqe_solver import Ansatze
from qsdk.problem_decomposition import DMETProblemDecomposition
from qsdk.problem_decomposition.electron_localization import iao_localization, meta_lowdin_localization

H4_RING = [['H', [0.7071067811865476,   0.0,                 0.0]],
           ['H', [0.0,                  0.7071067811865476,  0.0]],
           ['H', [-1.0071067811865476,  0.0,                 0.0]],
           ['H', [0.0,                 -1.0071067811865476,  0.0]]]


#@unittest.skip("DMET-VQE integration not available yet")
class DMETVQETest(unittest.TestCase):

    def test_h4ring_vqe_uccsd(self):
        """ DMET on H4 ring with fragment size one, using VQE-UCCSD """

        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        # Run DMET
        dmet = DMETProblemDecomposition()
        dmet.electron_localization_method = meta_lowdin_localization

        fragment_atoms = [1, 1, 1, 1]
        fragment_solvers = ['vqe', 'ccsd', 'ccsd', 'ccsd']
        energy = dmet.simulate(mol, fragment_atoms, fragment_solvers=fragment_solvers)

        self.assertAlmostEqual(energy, -1.9916120594, delta=1e-3)

    @unittest.skip("Newton solver fails to converge because of something related to convergence. "
                   "Changing tolerance or initial guess may help")
    def test_h4ring_vqe_uccsd_qiskit_size2(self):
        """
        DMET on H4 ring with fragment size two, using VQE-UCCSD backend
        from Qiskit.
        """

        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        # Initialize VQE object with Qiskit backend
        vqe = VQESolver()
        vqe.hardware_backend_type = QiskitParametricSolver
        vqe.ansatz_type = QiskitParametricSolver.Ansatze.UCCSD

        # Run DMET
        dmet = DMETProblemDecomposition()
        dmet.electron_localization_method = meta_lowdin_localization
        dmet.electronic_structure_solver = vqe
        energy_vqe = dmet.simulate(mol, [2,2])

        self.assertAlmostEqual(energy_vqe, -1.9916120594, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
