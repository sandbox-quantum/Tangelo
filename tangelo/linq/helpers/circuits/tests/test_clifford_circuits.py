import unittest

from math import pi as pi
from tangelo.linq import Gate, Circuit

from tangelo.linq.helpers.circuits import get_clifford_circuit, decompose_gate_to_cliffords

gates_1 = [Gate("H", 2), Gate("CNOT", 1, control=0),
           Gate("RZ", 0, parameter = -0.1 , is_variational=True),
           Gate("CNOT", 2, control = 1),
           Gate("RX", 0, parameter = -2.9 * pi, is_variational=True),
           Gate("RY", 0, parameter = 9.2*pi , is_variational=True),
           Gate("PHASE", 2, parameter = 0.1 , is_variational=True)]

circuit_1 = Circuit()
for gate in gates_1:
    circuit_1.add_gate(gate)

gates_ref = [Gate("H", 2), Gate("CNOT", 1, control=0),
            Gate("RZ", 0, parameter = 0 , is_variational=True),
            Gate("CNOT", 2, control = 1),
            Gate("RX", 0, parameter = -pi, is_variational=True),
            Gate("RY", 0, parameter = pi , is_variational=True),
            Gate("PHASE", 2, parameter = 0 , is_variational=True)]

circuit_ref = Circuit()
for gate in gates_ref:
    circuit_ref.add_gate(gate)

class CliffordCircuitTest(unittest.TestCase):

    def test_get_clifford_circuit(self):
        cliff_circuit = get_clifford_circuit(circuit_1)
        self.assertEqual(cliff_circuit, circuit_ref)


   # def test_decompose_clifford_gate(self):
        """Test initializing and uncomputing circuits with cirq lsq_first order"""
        #im = get_backend("cirq")

      #  np.testing.assert_array_almost_equal(clifford_state, ref_state)