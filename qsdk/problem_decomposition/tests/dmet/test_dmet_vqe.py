import unittest

from pyscf import gto

from qsdk.problem_decomposition.dmet.dmet_problem_decomposition import Localization, DMETProblemDecomposition

H4_RING = [['H', [0.7071067811865476,   0.0,                 0.0]],
           ['H', [0.0,                  0.7071067811865476,  0.0]],
           ['H', [-1.0071067811865476,  0.0,                 0.0]],
           ['H', [0.0,                 -1.0071067811865476,  0.0]]]


class DMETVQETest(unittest.TestCase):

    def test_h4ring_vqe_uccsd(self):
        """ DMET on H4 ring with fragment size one, using VQE-UCCSD """

        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": ['vqe', 'ccsd', 'ccsd', 'ccsd'],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        # Run DMET
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        energy = dmet.simulate()

        self.assertAlmostEqual(energy, -1.9916120594, delta=1e-3)

    def test_h4ring_vqe_jw_ressources(self):
        """ Resources estimation on H4 ring (JW). """

        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": "vqe",
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        ref_resources = [{'qubit_hamiltonian_terms': 15, 
                          'circuit_width': 4, 
                          'circuit_gates': 158, 
                          'circuit_2qubit_gates': 64, 
                          'circuit_var_gates': 12, 
                          'vqe_variational_parameters': 2}] * 4

        # Run DMET
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        resources = dmet.get_resources()

        self.assertEqual(resources, ref_resources)

    def test_h4ring_vqe_bk_ressources(self):
        """ Resources estimation on H4 ring (BK). """

        mol = gto.Mole()
        mol.atom = H4_RING
        mol.basis = "minao"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        vqe_options = {"qubit_mapping": "bk"}

        opt_dmet = {"molecule": mol,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": "vqe",
                    "electron_localization": Localization.meta_lowdin,
                    "solvers_options": vqe_options,
                    "verbose": False
                    }

        ref_resources = [{'qubit_hamiltonian_terms': 15, 
                          'circuit_width': 4, 
                          'circuit_gates': 107, 
                          'circuit_2qubit_gates': 46, 
                          'circuit_var_gates': 12, 
                          'vqe_variational_parameters': 2}] * 4

        # Run DMET
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        resources = dmet.get_resources()

        self.assertEqual(resources, ref_resources)


if __name__ == "__main__":
    unittest.main()
