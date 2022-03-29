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

"""Simple tests for input to mapping_transform.py. Presently, mostly using
wrappers on openfermion code, so pretty minimal testing here.
"""

import unittest
import numpy as np

from openfermion.linalg import eigenspectrum
from openfermion.linalg.sparse_tools import qubit_operator_sparse

from tangelo.toolboxes.operators import QubitOperator, FermionOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping, make_up_then_down


class MappingTest(unittest.TestCase):

    def test_bk(self):
        """Check output from Bravyi-Kitaev transformation."""
        bk_operator = QubitOperator(((0, "Z"), (1, "Y"), (2, "X")), -0.25j)
        bk_operator += QubitOperator(((0, "Z"), (1, "Y"), (2, "Y")), -0.25)
        bk_operator += QubitOperator(((1, "X"), (2, "X")), -0.25)
        bk_operator += QubitOperator(((1, "X"), (2, "Y")), 0.25j)
        bk_operator += QubitOperator(((0, "X"), (1, "Y"), (2, "Z")), -0.125j)
        bk_operator += QubitOperator(((0, "X"), (1, "X"), (3, "Z")), -0.125)
        bk_operator += QubitOperator(((0, "Y"), (1, "Y"), (2, "Z")), -0.125)
        bk_operator += QubitOperator(((0, "Y"), (1, "X"), (3, "Z")), 0.125j)

        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        n_spinorbitals = 4
        qubit = fermion_to_qubit_mapping(fermion, mapping="BK", n_spinorbitals=n_spinorbitals)
        self.assertEqual(qubit, bk_operator)

    def test_jw(self):
        """Check output from Jordan-Wigner transformation."""
        jw_operator = QubitOperator(((1, "Y"), (2, "X")), -0.25j)
        jw_operator += QubitOperator(((1, "Y"), (2, "Y")), -0.25)
        jw_operator += QubitOperator(((1, "X"), (2, "X")), -0.25)
        jw_operator += QubitOperator(((1, "X"), (2, "Y")), 0.25j)
        jw_operator += QubitOperator(((0, "Y"), (1, "Z"), (2, "Z"), (3, "X")), -0.125j)
        jw_operator += QubitOperator(((0, "Y"), (1, "Z"), (2, "Z"), (3, "Y")), 0.125)
        jw_operator += QubitOperator(((0, "X"), (1, "Z"), (2, "Z"), (3, "X")), 0.125)
        jw_operator += QubitOperator(((0, "X"), (1, "Z"), (2, "Z"), (3, "Y")), 0.125j)

        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        qubit = fermion_to_qubit_mapping(fermion, mapping="JW")
        self.assertEqual(qubit, jw_operator)

    def test_scbk(self):
        """Check output from symmetry-conserving Bravyi-Kitaev transformation.
        """
        scbk_operator = QubitOperator(((0, "Y"),), 1.j)

        fermion = FermionOperator(((2, 0), (0, 1)), 1.) + FermionOperator(((0, 0), (2, 1)), -1.)

        qubit = fermion_to_qubit_mapping(fermion, mapping="SCBK", n_spinorbitals=4, n_electrons=2)
        self.assertEqual(qubit, scbk_operator)

    def test_jkmn(self):
        """Check output from JKMN transform"""
        jkmn_operator = QubitOperator(((0, "X"), (2, "Z"), (3, "X")), 0.125)
        jkmn_operator += QubitOperator(((0, "X"), (2, "Z"), (3, "Y")), 0.125j)
        jkmn_operator += QubitOperator(((0, "Y"), (1, "Z"), (3, "X")), -0.125j)
        jkmn_operator += QubitOperator(((0, "Y"), (1, "Z"), (3, "Y")), 0.125)
        jkmn_operator += QubitOperator(((0, "Z"), (1, "X"), (2, "X")), 0.25j)
        jkmn_operator += QubitOperator(((0, "Z"), (1, "X"), (2, "Y")), 0.25)
        jkmn_operator += QubitOperator(((0, "Z"), (1, "Y"), (2, "X")), -0.25)
        jkmn_operator += QubitOperator(((0, "Z"), (1, "Y"), (2, "Y")), 0.25j)

        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)

        qubit = fermion_to_qubit_mapping(fermion, mapping="JKMN", n_spinorbitals=4)
        self.assertEqual(qubit, jkmn_operator)

    def test_scbk_invalid(self):
        """Check if fermion operator fails to conserve number parity or spin
        parity. In either case, scBK is not an appropriate mapping.
        """
        # excitation violating number and spin parity
        fermion = FermionOperator((1, 1), 1.0)
        with self.assertRaises(ValueError):
            fermion_to_qubit_mapping(fermion, mapping="SCBK", n_spinorbitals=2, n_electrons=1)
        # excitation violating spin parity
        fermion = FermionOperator(((0, 1), (1, 0)), 1.0)
        with self.assertRaises(ValueError):
            fermion_to_qubit_mapping(fermion, mapping="SCBK", n_spinorbitals=2, n_electrons=1)

    def test_handle_invalid_mapping(self):
        """Test that error is handled if invalid mapping is requested."""
        fermion = FermionOperator(((1, 0), (2, 1)), 1.0) + FermionOperator(((0, 1), (3, 0)), 0.5)
        with self.assertRaises(ValueError):
            fermion_to_qubit_mapping(fermion, mapping="invalid_mapping")

    def test_eigen(self):
        """Test that all encodings of the operator have the same ground state
        energy.
        """
        fermion = FermionOperator(((1, 0), (3, 1)), 1.0) + FermionOperator(((3, 0), (1, 1)), -1.0)
        ground = np.imag(eigenspectrum(fermion)).min()

        jw_operator = fermion_to_qubit_mapping(fermion, mapping="JW")
        bk_operator = fermion_to_qubit_mapping(fermion, mapping="BK", n_spinorbitals=4)
        scbk_operator = fermion_to_qubit_mapping(fermion, mapping="SCBK", n_spinorbitals=4, n_electrons=2)
        jkmn_operator = fermion_to_qubit_mapping(fermion, mapping="JKMN", n_spinorbitals=4)

        jw_ground = np.linalg.eigvalsh(qubit_operator_sparse(jw_operator).todense()).min()
        bk_ground = np.linalg.eigvalsh(qubit_operator_sparse(bk_operator, n_qubits=4).todense()).min()
        scbk_ground = np.linalg.eigvalsh(qubit_operator_sparse(scbk_operator, n_qubits=2).todense()).min()
        jkmn_ground = np.linalg.eigvalsh(qubit_operator_sparse(jkmn_operator, n_qubits=4).todense()).min()

        self.assertAlmostEqual(ground, jw_ground, places=15)
        self.assertAlmostEqual(ground, bk_ground, places=15)
        self.assertAlmostEqual(ground, scbk_ground, places=15)
        self.assertAlmostEqual(ground, jkmn_ground, places=15)

    def test_spin_order(self):
        """Test that re-ordering of spin-orbitals from alternating up down to
        all up, then all down produces expected result.
        """
        input_operator = FermionOperator(((1, 1), (3, 0), (4, 1), (6, 0)), 1.0) \
                         + FermionOperator(((2, 0), (5, 0), (5, 1), (7, 0)), 1.0)
        expected = FermionOperator(((4, 1), (5, 0), (2, 1), (3, 0)), 1.0) \
                   + FermionOperator(((1, 0), (6, 0), (6, 1), (7, 0)), 1.0)
        reordered_operator = make_up_then_down(input_operator, 8)
        self.assertEqual(expected, reordered_operator)

    def test_eigen_spin_reorder(self):
        """Test that all encodings of the operator have the same ground state
        energy, after re-ordering spins. Skip scBK here as this transformation
        forces spin-reordering.
        """
        fermion = FermionOperator(((1, 0), (3, 1)), 1.0) + FermionOperator(((3, 0), (1, 1)), -1.0)
        ground = np.imag(eigenspectrum(fermion)).min()

        jw_operator = fermion_to_qubit_mapping(fermion, mapping="JW", n_spinorbitals=4, up_then_down=True)
        bk_operator = fermion_to_qubit_mapping(fermion, mapping="BK", n_spinorbitals=4, up_then_down=True)
        jkmn_operator = fermion_to_qubit_mapping(fermion, mapping="JKMN", n_spinorbitals=4, up_then_down=True)

        jw_ground = np.linalg.eigvalsh(qubit_operator_sparse(jw_operator).todense()).min()
        bk_ground = np.linalg.eigvalsh(qubit_operator_sparse(bk_operator, n_qubits=4).todense()).min()
        jkmn_ground = np.linalg.eigvalsh(qubit_operator_sparse(jkmn_operator, n_qubits=4).todense()).min()

        self.assertEqual(ground, jw_ground)
        self.assertEqual(ground, bk_ground)
        self.assertEqual(ground, jkmn_ground)

    def test_scbk_reorder(self):
        """scBK forces spin-orbital ordering to all up then all down. Check that
        the qubit Hamiltonian returned is the same whether the user passes a
        FermionOperator with this ordering, or not.
        """
        fermion = FermionOperator(((2, 0), (0, 1)), 1.) + FermionOperator(((0, 0), (2, 1)), -1.)
        scBK_reordered = fermion_to_qubit_mapping(fermion_operator=fermion,
                                                  mapping="scBK",
                                                  n_spinorbitals=4,
                                                  n_electrons=2,
                                                  up_then_down=True)
        scBK_notreordered = fermion_to_qubit_mapping(fermion_operator=fermion,
                                                     mapping="scBK",
                                                     n_spinorbitals=4,
                                                     n_electrons=2,
                                                     up_then_down=False)
        self.assertEqual(scBK_reordered, scBK_notreordered)


if __name__ == "__main__":
    unittest.main()
