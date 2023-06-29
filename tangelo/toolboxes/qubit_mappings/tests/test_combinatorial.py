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
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings import combinatorial
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

        self.assertEqual(quop_00, QubitOperator("", 0.5) + QubitOperator("Z0", 0.5))
        self.assertEqual(quop_01, QubitOperator("X0", 0.5*2.22) + QubitOperator("Y0", 0.5j*2.22))
        self.assertEqual(quop_10, QubitOperator("X0", 0.5*3j) + QubitOperator("Y0", -0.5j*3j))
        self.assertEqual(quop_11, QubitOperator("", 0.5*(4+1j)) + QubitOperator("Z0", -0.5*(4+1j)))

    def test_element_to_qubitop_4by4(self):
        """Test the mapping of a 4x4 matrix element to a qubit operator."""

        quop_01_00 = element_to_qubitop(2, 1, 0, 5.)

        ref_op = QubitOperator("X0", 0.25) + QubitOperator("Y0", -0.25j) + \
            QubitOperator("X0 Z1", 0.25) + QubitOperator("Y0 Z1", -0.25j)
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
        """Test the function that applies an operator a^{\dagger}_j a_i."""

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
        qubit_op = combinatorial(H_ferm, mol_H2_sto3g.n_active_mos,
            mol_H2_sto3g.n_active_electrons)

        ref_qubit_op = QubitOperator("", -0.3399536)
        ref_qubit_op += QubitOperator("Y0 Y1", -0.181288)
        ref_qubit_op += QubitOperator("Z0", -0.3939836)
        ref_qubit_op += QubitOperator("Z0 Z1", 0.0112365)
        ref_qubit_op += QubitOperator("Z1", -0.3939836)

        self.assertTrue(qubit_op.isclose(ref_qubit_op, tol=1e-4))


if __name__ == "__main__":
    unittest.main()
