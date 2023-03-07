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

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings import combinatorial


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
