import unittest

from math import pi as pi
from tangelo.linq import Gate, Circuit

from tangelo.linq.helpers.circuits.clifford_circuits import decompose_gate_to_cliffords

gates_decomp = Gate("RX", target=0, parameter=pi / 2)#, Gate("RY", target=1, parameter=-pi / 2)]

gates_ref_decomp = [Gate("SDAG", 0), Gate("H", 0), Gate("SDAG", 0)]
                    #Gate("Z", 1), Gate("H", 1)]

class CliffordCircuitTest(unittest.TestCase):

    def test_decompose_gate_to_cliffords(self):
        self.assertEqual(gates_ref_decomp, decompose_gate_to_cliffords(gates_decomp))
    #def test_non_clifford_gates(self):

