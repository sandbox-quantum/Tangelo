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
import os

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2_sto3g, xyz_H2O
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list


H2_list = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
H2_string = """
H       0.0        0.0        0.0
H       0.0        0.0        0.7414
"""
H2_file_xyz = os.path.dirname(os.path.abspath(__file__))+"/data/h2.xyz"

H2O_list = [("O", (0., 0., 0.11779)),
            ("H", (0., 0.75545, -0.47116)),
            ("H", (0., -0.75545, -0.47116))]


def atom_list_close(atom1, atom2, atol):

    for (a0, xyz0), (a1, xyz1) in zip(atom1, atom2):
        if a0 != a1:
            raise AssertionError(f"Atoms are not the same {a0} != {a1}")
        for x0, x1 in zip(xyz0, xyz1):
            if abs(x0 - x1) > atol:
                raise AssertionError(f"geometries for atom {a0} are different. {x0} != {x1}")


class CoordsTest(unittest.TestCase):

    def test_atoms_string_to_list(self):
        """Verify conversion from string to list format for atom coordinates."""
        assert(atom_string_to_list(H2_string) == H2_list)


class SecondQuantizedMoleculeTest(unittest.TestCase):

    def test_instantiate_H2(self):
        """Verify basic properties of molecule object through instantiation of
        MolecularData class.
        """

        molecule = SecondQuantizedMolecule(H2_list, 0, 0, "sto-3g")

        assert(molecule.elements == ["H"]*2)
        assert(molecule.basis == "sto-3g")
        assert(molecule.spin == 0)
        assert(molecule.q == 0)
        assert(molecule.n_electrons == 2)

        molecule = SecondQuantizedMolecule(H2_file_xyz, 0, 0, "sto-3g")
        assert(molecule.elements == ["H"]*2)
        assert(molecule.basis == "sto-3g")
        assert(molecule.spin == 0)
        assert(molecule.q == 0)
        assert(molecule.n_electrons == 2)
        atom_list_close(molecule.xyz, H2_list, 1.e-14)

    def test_freezing_orbitals(self):
        """Verify freezing orbitals functionalities."""

        no_freezing = SecondQuantizedMolecule(H2O_list, 0, 0, "sto-3g", frozen_orbitals=None)
        assert(no_freezing.active_occupied == [0, 1, 2, 3, 4])
        assert(no_freezing.active_virtual == [5, 6])
        assert(no_freezing.frozen_occupied == [])
        assert(no_freezing.frozen_virtual == [])

        freeze_with_int = SecondQuantizedMolecule(H2O_list, frozen_orbitals=3)
        assert(freeze_with_int.active_occupied == [3, 4])
        assert(freeze_with_int.active_virtual == [5, 6])
        assert(freeze_with_int.frozen_occupied == [0, 1, 2])
        assert(freeze_with_int.frozen_virtual == [])

        freeze_with_list = SecondQuantizedMolecule(H2O_list, frozen_orbitals=[0, 1, 2, 6])
        assert(freeze_with_list.active_occupied == [3, 4])
        assert(freeze_with_list.active_virtual == [5])
        assert(freeze_with_list.frozen_occupied == [0, 1, 2])
        assert(freeze_with_list.frozen_virtual == [6])

    def test_freezing_empty(self):
        """Verify freezing orbitals empty input."""

        # None should result in the same as nothing.
        none_as_frozen = SecondQuantizedMolecule(H2O_list, frozen_orbitals=None)
        assert(none_as_frozen.active_occupied == [0, 1, 2, 3, 4])
        assert(none_as_frozen.active_virtual == [5, 6])
        assert(none_as_frozen.frozen_occupied == [])
        assert(none_as_frozen.frozen_virtual == [])

        # An empty list should result in the same as nothing.
        empty_as_frozen = SecondQuantizedMolecule(H2O_list, frozen_orbitals=[])
        assert(empty_as_frozen.active_occupied == [0, 1, 2, 3, 4])
        assert(empty_as_frozen.active_virtual == [5, 6])
        assert(empty_as_frozen.frozen_occupied == [])
        assert(empty_as_frozen.frozen_virtual == [])

    def test_freezing_type_exception(self):
        """Verify freezing orbitals exceptions."""

        # Cases where the input is wrong type.
        with self.assertRaises(TypeError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals="3")
        with self.assertRaises(TypeError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=3.141592)
        with self.assertRaises(TypeError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=[0, 1, 2.2222, 3, 4, 5])

    def test_no_active_electron(self):
        """Verify if freezing all active orbitals fails."""

        # Cases where no active electron are remaining.
        with self.assertRaises(ValueError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=5)
        with self.assertRaises(ValueError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=[0, 1, 2, 3, 4, 5])

    def test_no_active_virtual(self):
        """Verify if freezing all virtual orbitals fails."""

        # Cases where no active virtual orbitals are remaining.
        with self.assertRaises(ValueError):
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=[5, 6])

    def test_all_active_orbitals_occupied_but_some_not_fully(self):
        """Verify that having all active orbitals occupied but only partially occupied is permitted"""
        try:
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=[0, 1, 2, 3, 6], spin=2)
        except ValueError:
            self.fail("Unexpected ValueError raised")

    def test_get_fermionic_hamiltonian(self):
        """Verify energy shift in molecular hamiltonian."""

        molecule = SecondQuantizedMolecule(H2O_list, frozen_orbitals=1)
        shift = molecule.fermionic_hamiltonian.constant
        self.assertAlmostEqual(shift, -51.47120372466002, delta=1e-6)

    def test_energy_from_rdms(self):
        """Verify energy computation from RDMs."""

        rdm1 = [[ 1.97453997e+00, -7.05987336e-17],
                [-7.05987336e-17,  2.54600303e-02]]

        rdm2 = [
            [[[ 1.97453997e+00, -7.96423130e-17],
              [-7.96423130e-17,  3.21234218e-33]],
             [[-7.96423130e-17, -2.24213843e-01],
              [ 0.00000000e+00,  9.04357944e-18]]],
            [[[-7.96423130e-17,  0.00000000e+00],
              [-2.24213843e-01,  9.04357944e-18]],
             [[ 3.21234218e-33,  9.04357944e-18],
              [ 9.04357944e-18,  2.54600303e-02]]]
            ]

        e_rdms = mol_H2_sto3g.energy_from_rdms(rdm1, rdm2)
        self.assertAlmostEqual(e_rdms, -1.1372701, delta=1e-5)

    def test_symmetry_label(self):
        """Verify that the symmetry labels are correct when symmetry=True or symmetry="C2v"."""
        mo_symm_labels = ["A1", "A1", "B1", "A1", "B2", "A1", "B1"]
        mo_symm_ids = [0, 0, 2, 0, 3, 0, 2]

        molecule = SecondQuantizedMolecule(xyz=xyz_H2O, q=0, spin=0, symmetry=True, basis="sto-3g")
        assert(mo_symm_labels == molecule.mo_symm_labels)
        assert(mo_symm_ids == molecule.mo_symm_ids)

        molecule = SecondQuantizedMolecule(xyz=xyz_H2O, q=0, spin=0, symmetry="C2v", basis="sto-3g")
        assert(mo_symm_labels == molecule.mo_symm_labels)
        assert(mo_symm_ids == molecule.mo_symm_ids)

    def test_ecp(self):
        """Verify that the number of electrons is reduced when ecp is called."""

        molecule = SecondQuantizedMolecule(xyz="Yb", q=0, spin=0, basis="crenbl", ecp="crenbl")
        # "Yb" has 70 electrons but the ecp reduces this to 16
        assert(molecule.n_active_electrons == 16)
        assert(molecule.n_active_mos == 96)

        molecule = SecondQuantizedMolecule(xyz="Cu", q=0, spin=1, basis="cc-pvdz", ecp="crenbl",
                                           frozen_orbitals=list(range(8)))
        # "Cu" has 29 electrons but the ecp reduces this to 19. The active electrons are 19 - 8 * 2 = 3
        assert(molecule.n_active_electrons == 3)
        assert(molecule.n_active_mos == 35)


if __name__ == "__main__":
    unittest.main()
