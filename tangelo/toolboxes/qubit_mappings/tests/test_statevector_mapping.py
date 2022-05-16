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

"""Tests for statevector mapping methods, which carry a numpy array indicating
fermionic occupation of reference state into qubit representation.
"""

import unittest
import numpy as np

from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector, vector_to_circuit
from tangelo.molecule_library import mol_H4_sto3g, mol_H4_cation_sto3g
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.linq import Simulator

sim = Simulator()


class TestVector(unittest.TestCase):

    def test_jw_value(self):
        """Check that Jordan-Wigner mapping returns correct vector, for both
        default spin orderings.
        """
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_updown = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        output_jw = get_vector(vector.size, sum(vector), mapping="jw", up_then_down=False)
        output_jw_updown = get_vector(vector.size, sum(vector), mapping="jw", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector - output_jw), 0.0)
        self.assertEqual(np.linalg.norm(vector_updown - output_jw_updown), 0.0)

    def test_bk_value(self):
        """Check that Bravyi-Kitaev mapping returns correct vector, for both
        default spin orderings.
        """
        vector = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        vector_bk = np.array([1, 0, 1, 0, 0, 0, 0, 0])
        vector_bk_updown = np.array([1, 0, 0, 0, 1, 0, 0, 0])

        output_bk = get_vector(vector.size, sum(vector), mapping="bk", up_then_down=False)
        output_bk_updown = get_vector(vector.size, sum(vector), mapping="bk", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector_bk - output_bk), 0.0)
        self.assertEqual(np.linalg.norm(vector_bk_updown - output_bk_updown), 0.0)

    def test_scbk_value(self):
        """Check that symmetry-conserving Bravyi-Kitaev mapping returns correct
        vector.
        """
        vector = np.array([1, 0, 0, 1, 0, 0])

        output_bk = get_vector(8, 4, mapping="SCBK", up_then_down=True)
        self.assertEqual(np.linalg.norm(vector - output_bk), 0.0)

    def test_circuit_build(self):
        """Check circuit width and size (number of X gates)."""
        vector = np.array([1, 1, 1, 1, 0, 0, 1, 1])
        circuit = vector_to_circuit(vector)
        self.assertEqual(circuit.size, sum(vector))
        self.assertEqual(circuit.width, vector.size)

    def test_all_same_energy_mol_H4_sto3g(self):
        """Check that all mappings return statevectors that have the same energy expectation
        for an even number of electrons and various spins"""
        ferm_op = mol_H4_sto3g.fermionic_hamiltonian
        qu_op_bk = fermion_to_qubit_mapping(ferm_op,
                                            "BK",
                                            mol_H4_sto3g.n_active_sos,
                                            mol_H4_sto3g.n_active_electrons,
                                            up_then_down=True)
        qu_op_jw = fermion_to_qubit_mapping(ferm_op,
                                            "JW",
                                            mol_H4_sto3g.n_active_sos,
                                            mol_H4_sto3g.n_active_electrons,
                                            up_then_down=True)
        qu_op_jkmn = fermion_to_qubit_mapping(ferm_op,
                                              "JKMN",
                                              mol_H4_sto3g.n_active_sos,
                                              mol_H4_sto3g.n_active_electrons,
                                              up_then_down=True)

        # Test for spin 0, 2, and 4
        for spin in range(3):
            vector_bk = get_vector(mol_H4_sto3g.n_active_sos,
                                   mol_H4_sto3g.n_active_electrons,
                                   mapping="BK",
                                   up_then_down=True,
                                   spin=spin*2)
            vector_scbk = get_vector(mol_H4_sto3g.n_active_sos,
                                     mol_H4_sto3g.n_active_electrons,
                                     mapping="SCBK",
                                     up_then_down=True,
                                     spin=spin*2)
            vector_jw = get_vector(mol_H4_sto3g.n_active_sos,
                                   mol_H4_sto3g.n_active_electrons,
                                   mapping="JW",
                                   up_then_down=True,
                                   spin=spin*2)
            vector_jkmn = get_vector(mol_H4_sto3g.n_active_sos,
                                     mol_H4_sto3g.n_active_electrons,
                                     mapping="JKMN",
                                     up_then_down=True,
                                     spin=spin*2)
            circuit_bk = vector_to_circuit(vector_bk)
            circuit_scbk = vector_to_circuit(vector_scbk)
            circuit_jw = vector_to_circuit(vector_jw)
            circuit_jkmn = vector_to_circuit(vector_jkmn)

            qu_op_scbk = fermion_to_qubit_mapping(ferm_op,
                                                  'SCBK',
                                                  mol_H4_sto3g.n_active_sos,
                                                  mol_H4_sto3g.n_active_electrons,
                                                  up_then_down=True,
                                                  spin=spin*2)

            e_bk = sim.get_expectation_value(qu_op_bk, circuit_bk)
            e_scbk = sim.get_expectation_value(qu_op_scbk, circuit_scbk)
            e_jw = sim.get_expectation_value(qu_op_jw, circuit_jw)
            e_jkmn = sim.get_expectation_value(qu_op_jkmn, circuit_jkmn)
            self.assertAlmostEqual(e_bk, e_jw, places=7, msg=f"Failed for bk vs jw for spin={spin}")
            self.assertAlmostEqual(e_jw, e_scbk, places=7, msg=f"Failed for jw vs scbk for spin={spin}")
            self.assertAlmostEqual(e_scbk, e_jkmn, places=7, msg=f"Failed for jkmn vs scbk for spin={spin}")

    def test_all_same_energy_mol_H4_cation_sto3g(self):
        """Check that all mappings return statevectors that have the same energy expectation
        for an odd number of electrons and various spins"""
        ferm_op = mol_H4_cation_sto3g.fermionic_hamiltonian
        qu_op_bk = fermion_to_qubit_mapping(ferm_op,
                                            "BK",
                                            mol_H4_cation_sto3g.n_active_sos,
                                            mol_H4_cation_sto3g.n_active_electrons,
                                            up_then_down=True)
        qu_op_jw = fermion_to_qubit_mapping(ferm_op,
                                            "JW",
                                            mol_H4_cation_sto3g.n_active_sos,
                                            mol_H4_cation_sto3g.n_active_electrons,
                                            up_then_down=True)
        qu_op_jkmn = fermion_to_qubit_mapping(ferm_op,
                                              "JKMN",
                                              mol_H4_cation_sto3g.n_active_sos,
                                              mol_H4_cation_sto3g.n_active_electrons,
                                              up_then_down=True)
        # Test for spin 1 and 3
        for spin in range(2):
            vector_bk = get_vector(mol_H4_cation_sto3g.n_active_sos,
                                   mol_H4_cation_sto3g.n_active_electrons,
                                   mapping="BK",
                                   up_then_down=True,
                                   spin=spin*2+1)
            vector_scbk = get_vector(mol_H4_cation_sto3g.n_active_sos,
                                     mol_H4_cation_sto3g.n_active_electrons,
                                     mapping="SCBK",
                                     up_then_down=True,
                                     spin=spin*2+1)
            vector_jw = get_vector(mol_H4_cation_sto3g.n_active_sos,
                                   mol_H4_cation_sto3g.n_active_electrons,
                                   mapping="JW",
                                   up_then_down=True,
                                   spin=spin*2+1)
            vector_jkmn = get_vector(mol_H4_cation_sto3g.n_active_sos,
                                     mol_H4_cation_sto3g.n_active_electrons,
                                     mapping="JKMN",
                                     up_then_down=True,
                                     spin=spin*2+1)
            circuit_bk = vector_to_circuit(vector_bk)
            circuit_scbk = vector_to_circuit(vector_scbk)
            circuit_jw = vector_to_circuit(vector_jw)
            circuit_jkmn = vector_to_circuit(vector_jkmn)

            qu_op_scbk = fermion_to_qubit_mapping(ferm_op,
                                                  'SCBK',
                                                  mol_H4_cation_sto3g.n_active_sos,
                                                  mol_H4_cation_sto3g.n_active_electrons,
                                                  up_then_down=True,
                                                  spin=spin*2+1)

            e_bk = sim.get_expectation_value(qu_op_bk, circuit_bk)
            e_scbk = sim.get_expectation_value(qu_op_scbk, circuit_scbk)
            e_jw = sim.get_expectation_value(qu_op_jw, circuit_jw)
            e_jkmn = sim.get_expectation_value(qu_op_jkmn, circuit_jkmn)
            self.assertAlmostEqual(e_bk, e_jw, places=7, msg=f"Failed for bk vs jw for spin={spin}")
            self.assertAlmostEqual(e_jw, e_scbk, places=7, msg=f"Failed for jw vs scbk for spin={spin}")
            self.assertAlmostEqual(e_scbk, e_jkmn, places=7, msg=f"Failed for scbk vs jkmn for spin={spin}")


if __name__ == "__main__":
    unittest.main()
