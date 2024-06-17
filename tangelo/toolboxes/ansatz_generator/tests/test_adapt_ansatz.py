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
import math

from tangelo.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.linq import Gate, Circuit

f_op = FermionOperator("2^ 3^ 0 1") - FermionOperator("0^ 1^ 2 3")
qu_op = fermion_to_qubit_mapping(f_op, "jw")
qu_op.terms = {term: math.copysign(1.0, coeff.imag) for term, coeff in qu_op.terms.items()}


class ADAPTAnsatzTest(unittest.TestCase):

    def test_adaptansatz_init(self):
        """Verify behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2, spin=0)
        ansatz.build_circuit()

    def test_adaptansatz_adding(self):
        """Verify operator addition behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2, spin=0)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        self.assertEqual(ansatz.n_var_params, 1)
        self.assertEqual(ansatz._n_terms_operators, [8])

    def test_adaptansatz_set_var_params(self):
        """Verify variational parameter tuning behavior of ADAPTAnsatz class."""

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2, spin=0)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        ansatz.set_var_params([1.999])
        self.assertEqual(ansatz.var_params, [1.999])

        with self.assertRaises(ValueError):
            ansatz.set_var_params([1.999, 2.999])

    def test_adaptansatz_reference_state_circuit(self):
        """ Verify variational gate parameters are frozen in a reference state circuit"""

        # Create simple reference circuit:
        ref_circuit = Circuit([
            Gate('RY', i, parameter=1.0, is_variational=True)
            for i in range(4)
        ])

        # Create ADAPTAnsatz ansatz with reference circuit
        adapt_ansatz = ADAPTAnsatz(n_spinorbitals=2, n_electrons=2, spin=0,
                            ansatz_options=dict(reference_state=ref_circuit))

        adapt_ansatz.build_circuit()

        adapt_circ = adapt_ansatz.circuit
        adapt_circ_gates = list(adapt_circ)

        # Ensure gates are correctly prepended to circuit
        self.assertEqual(adapt_circ.width, 4)
        self.assertEqual(adapt_circ_gates[0].name, 'RY')

        # Ensure reference circuit gates were correctly converted to
        # non-variational gates
        self.assertFalse(adapt_circ_gates[0].is_variational)

        # Add qu_op to ansatz
        adapt_ansatz.add_operator(qu_op)

        # Check ansatz parameters
        self.assertEqual(adapt_ansatz.n_var_params, 1)
        self.assertEqual(adapt_ansatz._n_terms_operators, [8])

        adapt_circ = adapt_ansatz.circuit
        self.assertEqual(adapt_circ.width, 4)

        # Ensure reference circuit gates were correctly converted to non-variational gates
        # with the same name
        ref_circ_gate_names = [ gate.name for gate in ref_circuit ]
        adapt_circ_gate_names = [ gate.name for gate in adapt_circ ]
        adapt_circ_gate_variationals = [ gate.is_variational for gate in adapt_circ ]

        self.assertListEqual(ref_circ_gate_names, adapt_circ_gate_names[:ref_circuit.size])
        self.assertTrue(not any(adapt_circ_gate_variationals[:ref_circuit.size]))


if __name__ == "__main__":
    unittest.main()
