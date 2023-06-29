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

import numpy as np
from openfermion.linalg import qubit_operator_sparse
from openfermion.utils import load_operator

from tangelo.linq import Gate, Circuit, get_backend
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.operators.trim_trivial_qubits import trim_trivial_qubits, trim_trivial_operator, trim_trivial_circuit, is_bitflip_gate


pwd_this_test = os.path.dirname(os.path.abspath(__file__))

qb_ham = load_operator("H4_JW_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)

# Generate reference and test circuits using single qcc generator
ref_qcc_op = 0.2299941483397896 * 0.5 * QubitOperator("Y0 X1 X2 X3")
qcc_op = 0.2299941483397896 * 0.5 * QubitOperator("Y1 X3 X5 X7")

ref_mf_gates = [Gate("RX", 0, parameter=np.pi), Gate("X", 2)]

mf_gates = [
            Gate("RZ", 0, parameter=np.pi/2), Gate("RX", 0, parameter=3.14159),
            Gate("RX", 1, parameter=np.pi), Gate("RZ", 2, parameter=np.pi),
            Gate("X", 4), Gate("X", 5), Gate("Z", 6),
            Gate("RZ", 6, parameter=np.pi), Gate("RX", 8, parameter=-3*np.pi),
            Gate("X", 8), Gate("RZ", 9, parameter=np.pi), Gate("Z", 9)
           ]

ref_pauli_words_gates = sum((exp_pauliword_to_gates(pword, coef) for pword, coef in ref_qcc_op.terms.items()), start=[])
pauli_words_gates = sum((exp_pauliword_to_gates(pword, coef) for pword, coef in qcc_op.terms.items()), start=[])

ref_circ = Circuit(ref_mf_gates + ref_pauli_words_gates)
circ = Circuit(mf_gates + pauli_words_gates)

# Reference energy for H4 molecule with single QCC generator
ref_value = -1.8039875664891176

# Reference indices and states to be removed from system
ref_trim_states = {0: 1, 2: 0, 4: 1, 6: 0, 8: 0, 9: 0}

# Reference for which mf_gates are bitflip gates
ref_bool = [False, True, True, False, True, True, False, False, True, True, False, False]

sim = get_backend()


class TrimTrivialQubits(unittest.TestCase):
    def test_trim_trivial_operator(self):
        """ Test if trimming operator returns the correct eigenvalue """

        trimmed_operator = trim_trivial_operator(qb_ham, trim_states={key: ref_trim_states[key] for key in [0, 2, 4, 6]}, reindex=False)
        self.assertAlmostEqual(np.min(np.linalg.eigvalsh(qubit_operator_sparse(trimmed_operator).todense())), ref_value, places=5)

    def test_is_bitflip_gate(self):
        """ Test if bitflip gate function correctly identifies bitflip gates """
        self.assertEqual(ref_bool, [is_bitflip_gate(g) for g in mf_gates])

    def test_trim_trivial_circuit(self):
        """ Test if circuit trimming returns the correct circuit, states, and indices  """

        trimmed_circuit, trim_states = trim_trivial_circuit(circ)
        self.assertEqual(ref_circ._gates, trimmed_circuit._gates)
        self.assertEqual(ref_trim_states, trim_states)

    def test_trim_trivial_qubits(self):
        """ Test if trim trivial qubit function produces correct and compatible circuits and operators """

        trimmed_operator, trimmed_circuit = trim_trivial_qubits(qb_ham, circ)
        self.assertAlmostEqual(np.min(np.linalg.eigvalsh(qubit_operator_sparse(trimmed_operator).todense())), ref_value, places=5)
        self.assertEqual(ref_circ._gates, trimmed_circuit._gates)
        self.assertAlmostEqual(sim.get_expectation_value(trimmed_operator, trimmed_circuit), ref_value, places=5)

    def test_trim_trivial_qubits(self):
        """ Test if trim trivial qubit function produces correct and compatible circuits and operators """

        circ = Circuit(mf_gates + pauli_words_gates, n_qubits=12)
        trimmed_operator, trimmed_circuit = trim_trivial_qubits(qb_ham+QubitOperator("Z10"), circ)
        self.assertAlmostEqual(np.min(np.linalg.eigvalsh(qubit_operator_sparse(trimmed_operator).todense())), ref_value+1, places=5)
        self.assertEqual(ref_circ._gates, trimmed_circuit._gates)
        self.assertAlmostEqual(sim.get_expectation_value(trimmed_operator, trimmed_circuit), ref_value+1, places=5)


if __name__ == "__main__":
    unittest.main()
