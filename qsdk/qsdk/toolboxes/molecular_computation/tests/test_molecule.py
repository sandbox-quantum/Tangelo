import unittest

from qsdk import SecondQuantizedMolecule
from qsdk.toolboxes.molecular_computation.molecule import atom_string_to_list


H2_list = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
H2_string = """
H       0.0        0.0        0.0
H       0.0        0.0        0.7414
"""

H2O_list = [
    ("O", (0., 0., 0.11779)),
    ("H", (0., 0.75545, -0.47116)),
    ("H", (0., -0.75545, -0.47116))
]


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

    def test_get_fermionic_hamiltonian(self):
        """Verify energy shift in molecular hamiltonian."""

        molecule = SecondQuantizedMolecule(H2O_list, frozen_orbitals=1)
        shift = molecule.fermionic_hamiltonian.constant
        self.assertAlmostEqual(shift, -51.47120372466002, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
