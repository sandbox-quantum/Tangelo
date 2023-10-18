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

import os
import unittest

from openfermion import load_operator

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import QubitOperator as QOp
from tangelo.toolboxes.qubit_mappings import combinatorial
from tangelo.toolboxes.qubit_mappings.combinatorial import basis, conf_to_integer, one_body_op_on_state


path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'


class FunctionsCombinatorialTest(unittest.TestCase):

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
        """Test the mapping of H2 STO-3G to a combinatorial (qubit) Hamiltonian. 2-qubit problem. """

        fop = mol_H2_sto3g.fermionic_hamiltonian
        qop = combinatorial(fop, mol_H2_sto3g.n_active_mos, mol_H2_sto3g.n_active_electrons)

        ref_qop = (QOp("Z1", -0.3939836) + QOp("Y0 Y1", -0.181288) +
                   QOp("Z0", -0.3939836) + QOp("Z0 Z1", 0.0112365) + -0.3399536)

        self.assertTrue(qop.isclose(ref_qop, tol=1e-4))

    def test_combinatorial_H2O_sto3g(self):
        """Test the mapping of H2O STO-3G to a combinatorial (qubit) Hamiltonian. 9-qubit problem. """

        xyz_h2o = """
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        """

        mol = SecondQuantizedMolecule(xyz_h2o, q=0, spin=0, basis='sto-3g')
        qop = combinatorial(mol.fermionic_hamiltonian, mol.n_active_mos, mol.n_active_electrons)

        quop_ref = load_operator(file_name='comb_quop_h2o_sto3g.data', data_directory=path_data,
                                 plain_text=True)
        self.assertTrue(qop.isclose(quop_ref, tol=1e-4))


if __name__ == "__main__":
    unittest.main()
