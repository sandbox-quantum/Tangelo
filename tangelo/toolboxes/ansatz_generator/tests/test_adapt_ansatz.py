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

import unittest
import math

from tangelo.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping

f_op = FermionOperator("2^ 3^ 0 1") - FermionOperator("0^ 1^ 2 3")
qu_op = fermion_to_qubit_mapping(f_op, "jw")
qu_op.terms = {term: math.copysign(1.0, coeff.imag) for term, coeff in qu_op.terms.items()}


class ADAPTAnsatzTest(unittest.TestCase):

    def test_adaptansatz_init(self):
        """Verify behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

    def test_adaptansatz_adding(self):
        """Verify operator addition behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        self.assertEqual(ansatz.n_var_params, 1)
        self.assertEqual(ansatz._n_terms_operators, [8])

    def test_adaptansatz_set_var_params(self):
        """Verify variational parameter tuning behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        ansatz.set_var_params([1.999])
        self.assertEqual(ansatz.var_params, [1.999])

        with self.assertRaises(ValueError):
            ansatz.set_var_params([1.999, 2.999])


if __name__ == "__main__":
    unittest.main()
