# Copyright 2021 1QB Information Technologies Inc.
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

from qsdk.backendbuddy import Simulator
from qsdk.algorithms.projective.quantum_imaginary_time import QITESolver
from qsdk.molecule_library import mol_H2_sto3g, mol_H4_sto3g
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


class QITESolverTest(unittest.TestCase):

    def test_instantiation_qite(self):
        """Try instantiating VQESolver with basic input."""

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

#    def test_operator_expectation_qite(self):
#        """ A test of the operator_expectation function, using optimal parameters and exact simulator """
#
#        qite_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": 'jw'}
#        qite_solver = VQESolver(qite_options)
#        qite_solver.build()
#
#        # Test using var_params input and Qubit Hamiltonian
#        energy = qite_solver.operator_expectation(qite_solver.qubit_hamiltonian, var_params=[5.86665842e-06, 5.65317429e-02])
#        self.assertAlmostEqual(energy, -1.137270422018, places=6)
#
#        # Test using updated var_params and Fermion Hamiltonian
#        qite_solver.ansatz.update_var_params([5.86665842e-06, 5.65317429e-02])
#        energy = qite_solver.operator_expectation(mol_H2_sto3g.fermionic_hamiltonian)
#        self.assertAlmostEqual(energy, -1.137270422018, places=6)
#
#        # Test the three in place operators
#        n_electrons = qite_solver.operator_expectation('N')
#        self.assertAlmostEqual(n_electrons, 2, places=6)
#        spin_z = qite_solver.operator_expectation('Sz')
#        self.assertAlmostEqual(spin_z, 0, places=6)
#        spin2 = qite_solver.operator_expectation('S^2')
#        self.assertAlmostEqual(spin2, 0, places=6)

    def test_simulate_h2(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        qite_options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw",
                       "verbose": True}
        qite_solver = QITESolver(qite_options)
        qite_solver.build()

        energy = qite_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_mapping_BK(self):
        """Test that BK mapping recovers the expected result, to within 1e-6 Ha,
        for the example of H2 and MP2 initial guess.
        """
        qite_options = {"molecule": mol_H2_sto3g, "verbose": False,
                       "qubit_mapping": "bk"}

        qite_solver = QITESolver(qite_options)
        qite_solver.build()
        energy = qite_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_scBK(self):
        """Test that scBK mapping recovers the expected result, to within
        1e-6 Ha, for the example of H2 and MP2 initial guess.
        """
        qite_options = {"molecule": mol_H2_sto3g, "verbose": False,
                       "qubit_mapping": "scbk"}

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
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
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
