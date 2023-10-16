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

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import QubitOperator as QOp
from tangelo.toolboxes.qubit_mappings import combinatorial, combinatorial2, combinatorial3, combinatorial4
from tangelo.toolboxes.qubit_mappings.combinatorial import element_to_qubitop, basis, \
    conf_to_integer, one_body_op_on_state


class FunctionsCombinatorialTest(unittest.TestCase):

    def test_element_to_qubitop_2by2(self):
        """Test the mapping of a 2x2 matrix to qubit operators."""

        array = np.array([
            [1, 2.22],
            [3j, 4+1j]
        ])

        quop_00 = element_to_qubitop(1, 0, 0, array[0][0])
        quop_01 = element_to_qubitop(1, 0, 1, array[0][1])
        quop_10 = element_to_qubitop(1, 1, 0, array[1][0])
        quop_11 = element_to_qubitop(1, 1, 1, array[1][1])

        self.assertEqual(quop_00, QOp("", 0.5) + QOp("Z0", 0.5))
        self.assertEqual(quop_01, QOp("X0", 0.5*2.22) + QOp("Y0", 0.5j*2.22))
        self.assertEqual(quop_10, QOp("X0", 0.5*3j) + QOp("Y0", -0.5j*3j))
        self.assertEqual(quop_11, QOp("", 0.5*(4+1j)) + QOp("Z0", -0.5*(4+1j)))

    def test_element_to_qubitop_4by4(self):
        """Test the mapping of a 4x4 matrix element to a qubit operator."""

        quop_01_00 = element_to_qubitop(2, 1, 0, 5.)

        ref_op = QOp("X0", 0.25) + QOp("Y0", -0.25j) + QOp("X0 Z1", 0.25) + QOp("Y0 Z1", -0.25j)
        ref_op *= 5.

        self.assertEqual(quop_01_00, ref_op)

    def test_basis(self):
        """Test the basis function to construct a combinatorial set."""

        bs = basis(4, 1)

        self.assertEqual(bs.keys(), {(0,), (1,), (2,), (3,)})
        self.assertEqual(list(bs.values()), [0, 1, 2, 3])

    def test_conf_to_integer(self):
        """Test the mapping of an electronic configuration to a unique int."""

        self.assertEqual(conf_to_integer((0, 1), 4), 0)
        self.assertEqual(conf_to_integer((0, 2), 4), 1)
        self.assertEqual(conf_to_integer((0, 3), 4), 2)
        self.assertEqual(conf_to_integer((1, 2), 4), 3)
        self.assertEqual(conf_to_integer((1, 3), 4), 4)
        self.assertEqual(conf_to_integer((2, 3), 4), 5)

    def test_one_body_op_on_state(self):
        r"""Test the function that applies an operator a^{\dagger}_j a_i."""

        conf_in = (0, 1, 2)

        # Test case where i is in conf_in and not j (phase unchanged).
        conf_out_a, phase_a = one_body_op_on_state(((3, 1), (0, 0)), conf_in)
        self.assertEqual(conf_out_a, (1, 2, 3))
        self.assertEqual(phase_a, 1)

        # Test case where i is in conf_in and not j (phase changed).
        conf_out_b, phase_b = one_body_op_on_state(((3, 1), (1, 0)), conf_in)
        self.assertEqual(conf_out_b, (0, 2, 3))
        self.assertEqual(phase_b, -1)

        # Test case where i is not in conf_in.
        conf_out_c, phase_c = one_body_op_on_state(((4, 1), (3, 0)), conf_in)
        self.assertEqual(conf_out_c, ())
        self.assertEqual(phase_c, 0)

        # Test case where j is in conf_in.
        conf_out_d, phase_d = one_body_op_on_state(((2, 1), (1, 0)), conf_in)
        self.assertEqual(conf_out_d, ())
        self.assertEqual(phase_d, 0)


class CombinatorialTest(unittest.TestCase):

    def test_combinatorial_h2_sto3g(self):
        """Test the mapping of H2 STO-3G to a combinatorial (qubit) Hamiltonian."""

        H_ferm = mol_H2_sto3g.fermionic_hamiltonian
        qubit_op = combinatorial2(H_ferm, mol_H2_sto3g.n_active_mos,
            mol_H2_sto3g.n_active_electrons)

        ref_qubit_op = (QOp("Z1", -0.3939836) + QOp("Y0 Y1", -0.181288) +
                        QOp("Z0", -0.3939836) + QOp("Z0 Z1", 0.0112365) +
                        -0.3399536)

        print(qubit_op, "\n")
        print(ref_qubit_op)

        self.assertTrue(qubit_op.isclose(ref_qubit_op, tol=1e-4))

    def test_combinatorial_new(self):
        """Test the mapping of H2 STO-3G to a combinatorial (qubit) Hamiltonian."""

        from time import time
        from openfermion import load_operator
        from tangelo.toolboxes.operators import count_qubits
        from tangelo import SecondQuantizedMolecule
        from tangelo.molecule_library import mol_H2_sto3g, mol_H2_321g, mol_H2O_sto3g, mol_H2O_321g

        xyz_hf = """
        F 0.0000 0.0000 0.0000
        H 0.0000 0.0000 0.9168
        """

        xyz_lih = """
        Li 0.0000 0.0000 0.0000
        H  0.0000 0.0000 1.5949
        """

        xyz_h4 = """
        H  0.0000 0.0000 0.0000
        H  0.0000 1.0000 0.0000
        H  1.0000 0.0000 0.0000
        H  1.0000 1.0000 0.0000
        """

        atoms = xyz_hf
        basis = '3-21G'
        mol = mol_H2O_sto3g

        if not mol:
            mol = SecondQuantizedMolecule(atoms, q=0, spin=0, basis=basis,
                                    frozen_orbitals="frozen_core")

        fop = mol.fermionic_hamiltonian
        print(f'fop terms = {len(fop.terms)} \t {mol.n_active_mos} \t {mol.n_active_electrons}')

        t1 = time()
        qop1 = combinatorial4(fop, mol.n_active_mos, mol.n_active_electrons)
        t2 = time()
        print(f'c1 time elapsed {t2-t1}s (#terms = {len(qop1.terms)})')

        qop2 = combinatorial3(fop, mol.n_active_mos, mol.n_active_electrons)
        t3 = time()
        print(f'c2 time elapsed {t3-t2}s (#terms = {len(qop2.terms)})')

        # qop2 = load_operator("comb_quop_lih_fc_ccpvdz.data",
        #                      data_directory='.', plain_text=True)
        #print(qop1, qop2)
        # print(len(qop1.terms), len(qop2.terms))
        # print(count_qubits(qop1), count_qubits(qop2))
        # q = qop2-qop1
        # q.compress(abs_tol=1e-4)
        # print(type(q), q)
        #
        self.assertTrue(qop1.isclose(qop2, tol=1e-4))


if __name__ == "__main__":
    unittest.main()
