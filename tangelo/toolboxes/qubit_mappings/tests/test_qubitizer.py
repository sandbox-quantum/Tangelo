# Copyright SandboxAQ 2021-2024.
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
from tangelo.toolboxes.qubit_mappings import jordan_wigner
from tangelo.toolboxes.operators import QubitOperator


class QubitizerTest(unittest.TestCase):

    def test_qubit_hamiltonian_JW_h2(self):
        """Verify computation of the Jordan-Wigner Hamiltonian for the H2
        molecule."""

        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        # Obtained with Openfermion
        reference_terms = {(): -0.0988639693354571, ((0, "Z"),): 0.17119774903432955, ((1, "Z"),): 0.17119774903432958,
                           ((2, "Z"),): -0.22278593040418496, ((3, "Z"),): -0.22278593040418496,
                           ((0, "Z"), (1, "Z")): 0.16862219158920938, ((0, "Z"), (2, "Z")): 0.120544822053018,
                           ((0, "Z"), (3, "Z")): 0.165867024105892, ((1, "Z"), (2, "Z")): 0.165867024105892,
                           ((1, "Z"), (3, "Z")): 0.120544822053018, ((2, "Z"), (3, "Z")): 0.17434844185575687,
                           ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): -0.045322202052874,
                           ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): 0.045322202052874,
                           ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): 0.045322202052874,
                           ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): -0.045322202052874}
        reference_op = QubitOperator()
        reference_op.terms = reference_terms

        self.assertTrue(qubit_hamiltonian.isclose(reference_op, tol=1e-5))


if __name__ == "__main__":
    unittest.main()
