import unittest

from qsdk.toolboxes.molecular_computation.molecule import atom_string_to_list
from qsdk.molecule_library import mol_H2_sto3g, mol_H2O_sto3g

class FunctionsTest(unittest.TestCase):

    def test_atoms_string_to_list(self):
        """ Verify conversion from string to list format for atom coordinates """

        H2_list = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
        H2_string = """
        H       0.0        0.0        0.0
        H       0.0        0.0        0.7414
        """

        assert(atom_string_to_list(H2_string) == H2_list)


class SecondQuantizedMoleculeTest(unittest.TestCase):

    def test_instantiate_H2(self):
        """ Verify basic properties of molecule object through instantiation of MolecularData class """

        assert(mol_H2_sto3g.elements == ['H']*2)
        assert(mol_H2_sto3g.basis == 'sto-3g')
        assert(mol_H2_sto3g.spin == 0)
        assert(mol_H2_sto3g.q == 0)
        assert(mol_H2_sto3g.n_electrons == 2)

    def test_freezing_orbitals(self):
        """ Verify freezing orbitals functionalities """

        assert(mol_H2O_sto3g.active_occupied == [0, 1, 2, 3, 4])
        assert(mol_H2O_sto3g.active_virtual == [5, 6])
        assert(mol_H2O_sto3g.frozen_occupied == [])
        assert(mol_H2O_sto3g.frozen_virtual == [])

        freeze_with_int = mol_H2O_sto3g.freeze_mos(3, inplace=False)
        assert(freeze_with_int.active_occupied == [3, 4])
        assert(freeze_with_int.active_virtual == [5, 6])
        assert(freeze_with_int.frozen_occupied == [0, 1, 2])
        assert(freeze_with_int.frozen_virtual == [])

        freeze_with_list = mol_H2O_sto3g.freeze_mos([0, 1, 2, 6], inplace=False)
        assert(freeze_with_list.active_occupied == [3, 4])
        assert(freeze_with_list.active_virtual == [5])
        assert(freeze_with_list.frozen_occupied == [0, 1, 2])
        assert(freeze_with_list.frozen_virtual == [6])

    def test_freezing_empty(self):
        """ Verify freezing orbitals empty input """

        # An empty list should result in the same as nothing.
        empty_as_frozen = mol_H2O_sto3g.freeze_mos([], inplace=False)
        assert(empty_as_frozen.active_occupied == [0, 1, 2, 3, 4])
        assert(empty_as_frozen.active_virtual == [5, 6])
        assert(empty_as_frozen.frozen_occupied == [])
        assert(empty_as_frozen.frozen_virtual == [])

    def test_freezing_type_exception(self):
        """ Verify freezing orbitals exceptions """

        # Cases where the input is wrong type.
        with self.assertRaises(TypeError):
            mol_H2O_sto3g.freeze_mos("3", inplace=False)
        with self.assertRaises(TypeError):
            mol_H2O_sto3g.freeze_mos(3.141592, inplace=False)
        with self.assertRaises(TypeError):
            mol_H2O_sto3g.freeze_mos([0, 1, 2.2222, 3, 4, 5], inplace=False)

    def test_no_active_electron(self):
        """ Verify if freezing all active orbitals fails """

        # Cases where no active electron are remaining.
        with self.assertRaises(ValueError):
            mol_H2O_sto3g.freeze_mos(5, inplace=False)
        with self.assertRaises(ValueError):
            mol_H2O_sto3g.freeze_mos([0, 1, 2, 3, 4, 5], inplace=False)

    def test_no_active_virtual(self):
        """ Verify if freezing all virtual orbitals fails """

        # Cases where no active virtual orbitals are remaining.
        with self.assertRaises(ValueError):
            mol_H2O_sto3g.freeze_mos([5, 6], inplace=False)


    def test_get_fermionic_hamiltonian(self):
        """ Verify energy shift in molecular hamiltonian """

        molecule = mol_H2O_sto3g.freeze_mos(1, inplace=False)
        shift = molecule.fermionic_hamiltonian.constant
        self.assertAlmostEqual(shift, -51.47120372466002, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
