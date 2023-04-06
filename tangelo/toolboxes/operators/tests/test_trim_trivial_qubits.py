import os
import unittest

import numpy as np
from openfermion.linalg import qubit_operator_sparse
from openfermion.utils import load_operator

from tangelo.linq import Gate, Circuit, get_backend
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.operators.trim_trivial_qubits import *

pwd_this_test = os.path.dirname(os.path.abspath(__file__))

qb_ham=load_operator("H4_JW_spinupfirst.data", data_directory=pwd_this_test+"/data", plain_text=True)

qcc_op = 0.2299941483397896*0.5*QubitOperator("Y1 X3 X5 X7")
mf_gates =[Gate("RZ", 0, parameter=3.141592653589793), Gate("RX", 0, parameter=3.141592653589793),Gate("RX", 1, parameter=3.141592653589793), Gate("RZ", 2, parameter=3.141592653589793), Gate("X", 4), Gate("X", 5)]
pauli_words_gates = []
pauli_words = qcc_op.terms.items()
for i, (pauli_word, coef) in enumerate(pauli_words):
    pauli_words_gates += exp_pauliword_to_gates(pauli_word, coef)

circ = Circuit(mf_gates) + Circuit(pauli_words_gates)

ref_value=-1.8039875664891176
sim=get_backend()

class TrimTrivialQubits(unittest.TestCase):
    
    def test_trim_trivial_qubits(self):
        
        trimmed_operator, trimmed_circuit = trim_trivial_qubits(qb_ham,circ)
        self.assertAlmostEqual(np.min(np.linalg.eigvalsh(qubit_operator_sparse(trimmed_operator).todense())), ref_value, places=5)
        self.assertAlmostEqual(sim.get_expectation_value(trimmed_operator, trimmed_circuit), ref_value, places=5)

if __name__ == "__main__":
    unittest.main()