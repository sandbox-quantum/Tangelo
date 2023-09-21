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

import numpy as np
from openfermion import get_sparse_operator

from tangelo import SecondQuantizedMolecule
from tangelo.algorithms import FCISolver
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.molecule_library import mol_H2_321g, mol_Be_321g, mol_H4_cation_sto3g, mol_H4_sto3g, xyz_H4


class FCISolverTest(unittest.TestCase):

    def test_fci_h2(self):
        """Test FCISolver against result from reference implementation (H2)."""

        solver = FCISolver(mol_H2_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.1478300596229851, places=6)

        one_rdm, two_rdm = solver.get_rdm()
        self.assertAlmostEqual(energy, mol_H2_321g.energy_from_rdms(one_rdm, two_rdm), places=6)

    def test_fci_be(self):
        """Test FCISolver against result from reference implementation (Be)."""

        solver = FCISolver(mol_Be_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.531444379108095, places=6)

    def test_fci_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without
        first running a simulation.
        """

        solver = FCISolver(mol_H2_321g)
        self.assertRaises(RuntimeError, solver.get_rdm)

    def test_fci_openshell(self):
        """Test that the fci implementation for an openshell molecule is working properly"""

        solver = FCISolver(mol_H4_cation_sto3g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.6419373, places=6)

    def test_fci_be_frozen_core(self):
        """ Test FCISolver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = FCISolver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.530687987160581, places=6)

    def test_fci_H4_interior_frozen_orbitals(self):
        """ Test FCISolver against result from reference implementation with interior frozen orbitals
        """

        mol_H4_sto3g_freeze3 = mol_H4_sto3g.freeze_mos([1, 3, 4], inplace=False)

        solver = FCISolver(mol_H4_sto3g_freeze3)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.803792808, places=5)

        one_rdm, two_rdm = solver.get_rdm()
        self.assertAlmostEqual(energy, mol_H4_sto3g_freeze3.energy_from_rdms(one_rdm, two_rdm), places=5)

    def test_fci_H4_modified_mo_coeff(self):
        """ Test FCISolver against result from reference implementation with interior frozen orbitals by manually defining
        a swap operation.
        """

        mol = SecondQuantizedMolecule(xyz_H4, frozen_orbitals=[1, 3, 4])

        unitary_mat = np.array([[ 0.99799630, -0.05866044,  0.02358562,  0.00246221],
                                [ 0.05673690,  0.99556613,  0.07416274,  0.01135272],
                                [-0.02767097, -0.07212301,  0.99605136, -0.04375241],
                                [-0.00431649, -0.01432819,  0.04272342,  0.99897486]])
        # Modify orbitals
        mol.mo_coeff = unitary_mat.T@mol.mo_coeff

        op = get_sparse_operator(fermion_to_qubit_mapping(mol.fermionic_hamiltonian, "JW")).toarray()
        energy_op = np.linalg.eigh(op)[0][0]

        solver = FCISolver(mol)
        energy = solver.simulate()
        self.assertAlmostEqual(energy, -1.8057990, places=5)
        self.assertAlmostEqual(energy_op, -1.8057990, places=5)


if __name__ == "__main__":
    unittest.main()
