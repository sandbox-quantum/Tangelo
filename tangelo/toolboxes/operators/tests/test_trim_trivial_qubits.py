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

from openfermion.linalg import qubit_operator_sparse
from openfermion.utils import load_operator

from tangelo.linq import Gate, Circuit, get_backend
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.operators.trim_trivial_qubits import *

pwd_this_test = os.path.dirname(os.path.abspath(__file__))

qb_ham=load_operator("H4_JW_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)

qcc_op = 0.2299941483397896 * 0.5 * QubitOperator("Y1 X3 X5 X7")

mf_gates = [
Gate("RZ", 0, parameter=3.141592653589793),
Gate("RX", 0, parameter=3.14159),
Gate("RX", 1, parameter=3.141592653589793),
Gate("RZ", 2, parameter=3.141592653589793),
Gate("X", 4),
Gate("X", 5),
Gate("Z", 6),
Gate("RZ", 6, parameter=3.141592653589793),
Gate("RX", 8, parameter=3.141592653589793),
Gate("X", 8),
Gate("RZ", 9, parameter=3.141592653589793),
Gate("Z", 9)
]

pauli_words_gates = []
for pauli_word, coef in qcc_op.terms.items():
    pauli_words_gates += exp_pauliword_to_gates(pauli_word, coef)

circ = Circuit(mf_gates) + Circuit(pauli_words_gates)

ref_value = -1.8039875664891176

sim = get_backend()

class TrimTrivialQubits(unittest.TestCase):
    def test_trim_trivial_qubits(self):

        trimmed_operator, trimmed_circuit = trim_trivial_qubits(qb_ham,circ)
        self.assertAlmostEqual(np.min(np.linalg.eigvalsh(qubit_operator_sparse(trimmed_operator).todense())), ref_value, places=5)
        self.assertAlmostEqual(sim.get_expectation_value(trimmed_operator, trimmed_circuit), ref_value, places=5)

if __name__ == "__main__":
    unittest.main()