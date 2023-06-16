import unittest

from math import pi as pi
from tangelo.linq import Gate

from tangelo.linq.helpers.circuits.clifford_circuits import decompose_gate_to_cliffords

clifford_gate = Gate("RX", target=0, parameter=pi / 2)
non_clifford_gate = Gate("RZ", target=0, parameter=0.2)
clifford_decomposed = [Gate("SDAG", 0), Gate("H", 0), Gate("SDAG", 0)]


class CliffordCircuitTest(unittest.TestCase):

    def test_decompose_gate_to_cliffords(self):
        """Test if gate decomposition returns correct sequence"""

        self.assertEqual(clifford_decomposed, decompose_gate_to_cliffords(clifford_gate))

    def test_non_clifford_gates(self):
        """Test if non-clifford gate raises value error"""

        self.assertRaises(ValueError, decompose_gate_to_cliffords, non_clifford_gate)



