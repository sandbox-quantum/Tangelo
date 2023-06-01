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

import unittest
import os

import numpy as np
from openfermion.utils import load_operator

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2_sto3g, xyz_H2O
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list
from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

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


class MoleculeUtilitiesTest(unittest.TestCase):

    def test_atoms_string_to_list(self):
        """Verify conversion from string to list format for atom coordinates."""
        assert(atom_string_to_list(H2_string) == H2_list)


class SecondQuantizedMoleculeTest(unittest.TestCase):

    def test_instantiate_H2(self):
        """Verify basic properties of molecule object through instantiation of
        MolecularData class.
        """

        molecule = SecondQuantizedMolecule(H2_list, 0, 0, basis="sto-3g")

        assert(molecule.elements == ["H"]*2)
        assert(molecule.basis == "sto-3g")
        assert(molecule.spin == 0)
        assert(molecule.q == 0)
        assert(molecule.n_electrons == 2)

        molecule = SecondQuantizedMolecule(H2_file_xyz, 0, 0, basis="sto-3g")
        assert(molecule.elements == ["H"]*2)
        assert(molecule.basis == "sto-3g")
        assert(molecule.spin == 0)
        assert(molecule.q == 0)
        assert(molecule.n_electrons == 2)
        atom_list_close(molecule.xyz, H2_list, 1.e-14)

    def test_all_active_orbitals_occupied_but_some_not_fully(self):
        """Verify that having all active orbitals occupied but only partially occupied is permitted"""
        try:
            SecondQuantizedMolecule(H2O_list, frozen_orbitals=[0, 1, 2, 3, 6], spin=2)
        except ValueError:
            self.fail("Unexpected ValueError raised")

    def test_get_fermionic_hamiltonian(self):
        """Verify the construction of a fermionic Hamiltonian."""

        ferm_op = mol_H2_sto3g.fermionic_hamiltonian
        ref_ferm_op = load_operator("H2_ferm_op.data", data_directory=pwd_this_test+"/data", plain_text=True)
        self.assertEqual(ferm_op, ref_ferm_op)

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
        mo_symm_labels = ["A1", "A1", "B2", "A1", "B1", "A1", "B2"]
        mo_symm_ids = [0, 0, 3, 0, 2, 0, 3]

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

    def test_mo_coeff_setter(self):
        """Verify the dimension check in the mo_coeff setter."""

        molecule = SecondQuantizedMolecule(H2_list, 0, 0, basis="sto-3g")

        # Should work.
        dummy_mo_coeff = np.ones((2, 2))
        molecule.mo_coeff = dummy_mo_coeff

        # Should raise an AssertionError.
        bad_dummy_mo_coeff = np.ones((3, 3))
        with self.assertRaises(AssertionError):
            molecule.mo_coeff = bad_dummy_mo_coeff

        molecule = SecondQuantizedMolecule(H2_list, 0, 0, basis="sto-3g", uhf=True)

        # Should work.
        dummy_mo_coeff = [np.ones((2, 2))]*2
        molecule.mo_coeff = dummy_mo_coeff

        # Should raise an AssertionError.
        bad_dummy_mo_coeff = [np.ones((3, 3))]*2
        with self.assertRaises(AssertionError):
            molecule.mo_coeff = bad_dummy_mo_coeff

    def test_custom_solver_H2(self):
        """Verify that using a custom ESSolver that only stores molecular data functions correctly returns
        energy_from_rdms and fermion_hamiltonian
        """

        molecule = SecondQuantizedMolecule(H2_list, 0, 0, basis="sto-3g")

        core_constant, one_body_integrals, two_body_integrals = molecule.get_full_space_integrals()

        class IntegralSolverDummy(IntegralSolver):
            def set_physical_data(self, mol):
                mol.xyz = H2_list
                mol.n_electrons = 2
                mol.n_atoms = 2

            def compute_mean_field(self, sqmol):
                sqmol.mf_energy = molecule.mf_energy
                sqmol.mo_energies = molecule.mo_energies
                sqmol.mo_occ = molecule.mo_occ
                sqmol.n_mos = molecule.n_mos
                sqmol.n_sos = molecule.n_sos
                self.mo_coeff = molecule.mo_coeff.copy()

            def get_integrals(self, sqmol, mo_coeff=None):
                return core_constant, one_body_integrals, two_body_integrals

        molecule_dummy = SecondQuantizedMolecule(H2_list, 0, 0, IntegralSolverDummy(), basis="sto-3g", frozen_orbitals=[])

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

        e_rdms = molecule_dummy.energy_from_rdms(rdm1, rdm2)
        self.assertAlmostEqual(e_rdms, -1.1372701, delta=1e-5)

        ferm_op = molecule_dummy.fermionic_hamiltonian
        ref_ferm_op = load_operator("H2_ferm_op.data", data_directory=pwd_this_test+"/data", plain_text=True)
        self.assertEqual(ferm_op, ref_ferm_op)


if __name__ == "__main__":
    unittest.main()
