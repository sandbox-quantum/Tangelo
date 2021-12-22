# Copyright 2021 Good Chemsitry Company.
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

from tangelo.algorithms.projective.quantum_imaginary_time import QITESolver
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g
from tangelo.linq.noisy_simulation import NoiseModel


class QITESolverTest(unittest.TestCase):

    def test_instantiation_qite(self):
        """Try instantiating QITESolver with basic input."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw"}
        QITESolver(options)

    def test_instantiation_qite_incorrect_keyword(self):
        """Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "dummy": True}
        self.assertRaises(KeyError, QITESolver, options)

    def test_instantiation_qite_missing_molecule(self):
        """Instantiating with no molecule should return an error."""

        options = {"qubit_mapping": "jw"}
        self.assertRaises(ValueError, QITESolver, options)

    def test_simulate_h2_noisy(self):
        """Run QITE on H2 molecule with bk qubit mapping and an empty noise model for 1 cycle.
        Result should be lower than mean field energy.
        """

        backend_options = {"target": None, "n_shots": 10000, "noise_model": NoiseModel()}

        qite_options = {"molecule": mol_H2_sto3g, "qubit_mapping": "scbk",
                        "verbose": True, "backend_options": backend_options,
                        "max_cycles": 1, "up_then_down": True}
        qite_solver = QITESolver(qite_options)
        qite_solver.build()

        energy = qite_solver.simulate()
        self.assertTrue(energy < mol_H2_sto3g.mf_energy)

    def test_simulate_h2(self):
        """Run QITE on H2 molecule, with JW qubit mapping and exact simulator
        """

        qite_options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw",
                        "verbose": False, "up_then_down": True}
        qite_solver = QITESolver(qite_options)
        qite_solver.build()

        energy = qite_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_resources_h2(self):
        """Test get_resources funtion for QITE on H2 molecule, with JW qubit mapping.
        """

        qite_options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw",
                        "verbose": False}
        qite_solver = QITESolver(qite_options)
        qite_solver.build()
        resources = qite_solver.get_resources()
        self.assertEqual(resources["qubit_hamiltonian_terms"], 15)
        self.assertEqual(resources["pool_size"], 20)

    def test_mapping_BK(self):
        """Test that BK mapping recovers the expected result for the example of H2.
        """
        qite_options = {"molecule": mol_H2_sto3g, "verbose": False,
                        "qubit_mapping": "bk"}

        qite_solver = QITESolver(qite_options)
        qite_solver.build()
        energy = qite_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_JKMN(self):
        """Test that JKMN mapping recovers the expected result for the example of H2.
        """
        qite_options = {"molecule": mol_H2_sto3g, "verbose": False,
                        "qubit_mapping": "JKMN"}

        qite_solver = QITESolver(qite_options)
        qite_solver.build()
        energy = qite_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_scBK(self):
        """Test that scBK mapping recovers the expected result for the example of H2.
        """
        qite_options = {"molecule": mol_H2_sto3g, "verbose": False,
                        "qubit_mapping": "scbk", "up_then_down": True}

        qite_solver = QITESolver(qite_options)
        qite_solver.build()
        energy = qite_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_spin_reorder_equivalence(self):
        """Test that re-ordered spin input (all up followed by all down) returns
        the same optimized energy result for both JW and BK mappings.
        """
        qite_options = {"molecule": mol_H2_sto3g, "up_then_down": True,
                        "verbose": False, "qubit_mapping": "jw"}

        qite_solver_jw = QITESolver(qite_options)
        qite_solver_jw.build()
        energy_jw = qite_solver_jw.simulate()

        qite_options["qubit_mapping"] = "bk"
        qite_solver_bk = QITESolver(qite_options)
        qite_solver_bk.build()
        energy_bk = qite_solver_bk.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy_jw, energy_target, places=5)
        self.assertAlmostEqual(energy_bk, energy_target, places=5)

    def test_simulate_h4_frozen_orbitals(self):
        """Run QITE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator. First (occupied) and last (virtual)
        orbitals are frozen.
        """
        mol_H4_sto3g_frozen = mol_H4_sto3g.freeze_mos([0, 3], inplace=False)

        qite_options = {"molecule": mol_H4_sto3g_frozen, "qubit_mapping": "jw",
                        "verbose": False}
        qite_solver = QITESolver(qite_options)
        qite_solver.build()

        energy = qite_solver.simulate()
        self.assertAlmostEqual(energy, -1.8943598012229799, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
