import unittest
from pyscf import gto
from pyscf.gto.mole import Mole

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData, atom_string_to_list

H2_list = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
H2_string = """ 
H       0.0        0.0        0.0
H       0.0        0.0        0.7414
"""

mol = gto.Mole()
mol.atom = H2_string
mol.basis = "sto-3g"
mol.spin = 0
mol.build()

H2O_list = [('O', (0., 0., 0.11779)), 
            ('H', (0., 0.75545, -0.47116)),
            ('H', (0., -0.75545, -0.47116))
        ]

mol_h2o = gto.Mole()
mol_h2o.atom = H2O_list
mol_h2o.basis = "sto-3g"
mol_h2o.spin = 0
mol_h2o.build()


class MolecularDataTest(unittest.TestCase):

    def test_atoms_string_to_list(self):
        """ Verify conversion from string to list format for atom coordinates """
        assert(atom_string_to_list(H2_string) == H2_list)

    def test_instantiate_H2(self):
        """ Verify basic properties of molecule object through instantiation of MolecularData class """

        molecule = MolecularData(mol)

        assert(molecule.atoms == ['H']*2)
        assert(molecule.basis == 'sto-3g')
        assert(molecule.multiplicity == 1)
        assert(molecule.charge == 0)
        assert(molecule.n_electrons == 2)
        assert(molecule.protons == [1]*2)

    def test_run_pyscf_h2(self):
        """ Verify basic properties of molecule object through instantiation of MolecularData class """

        molecule = MolecularData(mol)
        self.assertAlmostEqual(molecule.hf_energy, -1.1166843870853396, delta=1e-8)
        self.assertAlmostEqual(molecule.fci_energy, -1.1372701746609026, delta=1e-8)
        self.assertAlmostEqual(molecule.cisd_energy, -1.1372701746609017, delta=1e-8)
        self.assertAlmostEqual(molecule.ccsd_energy, -1.1372703406409188, delta=1e-8)

    def test_freezing_orbitals(self):
        """ Verify freezing orbitals functionalities """

        no_freezing = MolecularData(mol_h2o)
        assert(no_freezing.active_occupied == [0, 1, 2, 3, 4])
        assert(no_freezing.active_virtual == [5, 6])
        assert(no_freezing.frozen_occupied == [])
        assert(no_freezing.frozen_virtual == [])

        freeze_with_int = MolecularData(mol_h2o, 3)
        assert(freeze_with_int.active_occupied == [3, 4])
        assert(freeze_with_int.active_virtual == [5, 6])
        assert(freeze_with_int.frozen_occupied == [0, 1, 2])
        assert(freeze_with_int.frozen_virtual == [])

        freeze_with_list = MolecularData(mol_h2o, [0, 1, 2, 6])
        assert(freeze_with_list.active_occupied == [3, 4])
        assert(freeze_with_list.active_virtual == [5])
        assert(freeze_with_list.frozen_occupied == [0, 1, 2])
        assert(freeze_with_list.frozen_virtual == [6])

    def test_freezing_exception(self):
        """ Verify freezing orbitals exceptions """

        # None should result in the same as nothing.
        none_as_frozen = MolecularData(mol_h2o, None)
        assert(none_as_frozen.active_occupied == [0, 1, 2, 3, 4])
        assert(none_as_frozen.active_virtual == [5, 6])
        assert(none_as_frozen.frozen_occupied == [])
        assert(none_as_frozen.frozen_virtual == [])

        # An empty list should result in the same as nothing.
        empty_as_frozen = MolecularData(mol_h2o, [])
        assert(empty_as_frozen.active_occupied == [0, 1, 2, 3, 4])
        assert(empty_as_frozen.active_virtual == [5, 6])
        assert(empty_as_frozen.frozen_occupied == [])
        assert(empty_as_frozen.frozen_virtual == [])

        # Cases where the input is wrong type.
        with self.assertRaises(TypeError):
            MolecularData(mol_h2o, "3")
        with self.assertRaises(TypeError):
            MolecularData(mol_h2o, 3.141592)
        with self.assertRaises(TypeError):
            MolecularData(mol_h2o, [0, 1, 2.2222, 3, 4, 5])

        # Cases where no active electron are remaining.
        with self.assertRaises(ValueError):
            MolecularData(mol_h2o, 5)
        with self.assertRaises(ValueError):
            MolecularData(mol_h2o, [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
