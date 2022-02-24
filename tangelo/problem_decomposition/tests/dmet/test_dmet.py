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
import numpy as np

from tangelo.molecule_library import mol_H4_doublecation_minao, mol_H4_doublecation_321g, mol_H10_321g, mol_H10_minao
from tangelo.problem_decomposition import DMETProblemDecomposition
from tangelo.problem_decomposition.dmet import Localization
from tangelo.algorithms.variational import VQESolver
from tangelo.toolboxes.molecular_computation.rdms import matricize_2rdm


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

    def test_build_with_atom_indices(self):
        """Tests if a mean field is recomputed when providing atom indices."""

        opt_dmet = {"molecule": mol_H4_doublecation_321g,
                    "fragment_atoms": [[0], [1], [2], [3]],
                    "fragment_solvers": "ccsd",
                    "electron_localization": Localization.iao,
                    "verbose": False
                    }

        solver = DMETProblemDecomposition(opt_dmet)
        solver.build()

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

    def test_retrieving_quantum_data(self):
        """Test if getting back a fragment gives the same RDMs."""

        opt_dmet = {"molecule": mol_H10_minao,
                    "fragment_atoms": [1]*10,
                    "fragment_solvers": ["vqe"] + ["ccsd"]*9,
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": True,
                    "solvers_options": [{"qubit_mapping": "scBK",
                                         "initial_var_params": "ones",
                                         "up_then_down": True,
                                         "verbose": False}] + [{}]*9,
                    }

        dmet_solver = DMETProblemDecomposition(opt_dmet)
        dmet_solver.build()

        # One shot loop with the optimal chemical potential.
        dmet_solver.n_iter = 0
        dmet_solver._oneshot_loop(-0.0000105903, save_results=True)

        ref_onerdm = [[1.99836090e+00, 6.91189485e-04],
                      [6.91189485e-04, 1.63910482e-03]]
        ref_twordm = [[[[ 1.99836064e+00,  7.11566876e-04],
                        [ 7.11566876e-04,  2.53371393e-07]],
                        [[ 7.11566876e-04, -5.72277575e-02],
                        [ 2.53371393e-07, -2.03773912e-05]]],
                        [[[ 7.11566876e-04,  2.53371393e-07],
                        [-5.72277575e-02, -2.03773912e-05]],
                        [[ 2.53371393e-07, -2.03773912e-05],
                        [-2.03773912e-05,  1.63885145e-03]]]]

        fragment, _, q_circuit = dmet_solver.quantum_fragments_data[0]

        vqe_solver = VQESolver({"molecule": fragment, "ansatz": q_circuit,
                                "qubit_mapping": "scBK"})
        vqe_solver.build()
        vqe_solver.simulate()

        onerdm, twordm = vqe_solver.get_rdm(vqe_solver.optimal_var_params)

        # Test traces of matrices
        n_elec = fragment.n_active_electrons
        n_orb = fragment.n_active_sos // 2
        self.assertAlmostEqual(np.trace(onerdm), n_elec, delta=1e-3,
                               msg="Trace of one_rdm does not match number of electrons")

        rho = matricize_2rdm(twordm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1), delta=1e-3,
                               msg="Trace of two_rdm does not match n_elec * (n_elec-1)")


if __name__ == "__main__":
    unittest.main()
