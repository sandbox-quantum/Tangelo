# Copyright 2023 Good Chemistry Company.
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
from itertools import combinations

import numpy as np

from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector, vector_to_circuit
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H4_sto3g, mol_H4_cation_sto3g, xyz_H2O
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.linq import get_backend

sim = get_backend()
spin_ind_transforms = ["BK", "JW", "JKMN"]
spin_dep_transforms = ["SCBK"]
all_transforms = spin_ind_transforms + spin_dep_transforms


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

    def test_all_same_energy_mol_H4_sto3g_and_mol_H2O_sto3g(self):
        """Check that all mappings return statevectors that have the same energy expectation
        for an even number of electrons and various spins. Molecules tested are H4, and H2O
        with frozen_orbitals=[0, 7] which failed previously for scbk"""
        mols = [mol_H4_sto3g, SecondQuantizedMolecule(xyz_H2O, 0, 0, basis="sto-3g", frozen_orbitals=[0, 7])]
        circuits = dict()
        qu_ops = dict()
        for mol in mols:
            ferm_op = mol.fermionic_hamiltonian
            qu_ops = dict()
            for transform in spin_ind_transforms:
                qu_ops[transform] = fermion_to_qubit_mapping(ferm_op,
                                                             transform,
                                                             mol.n_active_sos,
                                                             mol.n_active_electrons,
                                                             up_then_down=True)

            # Test for spin 0, 2, and 4
            for spin in range(3):
                # Get circuits for transforms not dependant on spin
                for transform in all_transforms:
                    vector = get_vector(mol.n_active_sos,
                                        mol.n_active_electrons,
                                        mapping=transform,
                                        up_then_down=True,
                                        spin=spin*2)
                    circuits[transform] = vector_to_circuit(vector)

                for transform in spin_dep_transforms:
                    qu_ops[transform] = fermion_to_qubit_mapping(ferm_op,
                                                                 'SCBK',
                                                                 mol.n_active_sos,
                                                                 mol.n_active_electrons,
                                                                 up_then_down=True,
                                                                 spin=spin*2)

                energies = dict()
                for transform in all_transforms:
                    energies[transform] = sim.get_expectation_value(qu_ops[transform], circuits[transform])

                for t1, t2 in combinations(all_transforms, 2):
                    self.assertAlmostEqual(energies[t1], energies[t2], places=7, msg=f"Failed for {t1} vs {t2} for spin={spin*2}")

    def test_all_same_energy_mol_H4_cation_sto3g(self):
        """Check that all mappings return statevectors that have the same energy expectation
        for an odd number of electrons and various spins"""
        ferm_op = mol_H4_cation_sto3g.fermionic_hamiltonian
        circuits = dict()
        qu_ops = dict()
        for transform in spin_ind_transforms:
            qu_ops[transform] = fermion_to_qubit_mapping(ferm_op,
                                                         transform,
                                                         mol_H4_cation_sto3g.n_active_sos,
                                                         mol_H4_cation_sto3g.n_active_electrons,
                                                         up_then_down=True)

        # Test for spin 1 and 3
        for spin in range(2):
            for transform in all_transforms:
                vector = get_vector(mol_H4_cation_sto3g.n_active_sos,
                                    mol_H4_cation_sto3g.n_active_electrons,
                                    mapping=transform,
                                    up_then_down=True,
                                    spin=spin*2+1)
                circuits[transform] = vector_to_circuit(vector)
            for transform in spin_dep_transforms:
                qu_ops[transform] = fermion_to_qubit_mapping(ferm_op,
                                                             transform,
                                                             mol_H4_cation_sto3g.n_active_sos,
                                                             mol_H4_cation_sto3g.n_active_electrons,
                                                             up_then_down=True,
                                                             spin=spin*2+1)

            energies = dict()
            for transform in all_transforms:
                energies[transform] = sim.get_expectation_value(qu_ops[transform], circuits[transform])

            for t1, t2 in combinations(all_transforms, 2):
                self.assertAlmostEqual(energies[t1], energies[t2], places=7, msg=f"Failed for {t1} vs {t2} for spin={spin*2}")


if __name__ == "__main__":
    unittest.main()
