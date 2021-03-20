import unittest
from pyscf import gto

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


if __name__ == "__main__":
    unittest.main()
