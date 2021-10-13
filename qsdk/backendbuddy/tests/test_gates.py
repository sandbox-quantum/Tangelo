# Copyright 2020-2021 1QB Information Technologies Inc.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#   http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from qsdk.backendbuddy import Gate


class TestGates(unittest.TestCase):
    """
        A test class to check that the circuit class is working properly, and processing the information
        of the gate objects it holds as expected.
    """

    def test_some_gates(self):
        """ Test that some basic gates can be invoked with a few different parameters, and that this information
        is printed as expected by the built-in __str__ method """

        # Create a Hadamard gate acting on qubit 2
        H_gate = Gate("H", 2)
        # Create a CNOT gate with control qubit 0 and target qubit 1
        CNOT_gate = Gate("CNOT", 1, 0)
        # Create a parameterized rotation on qubit 1 with angle 2 radians
        RX_gate = Gate("RX", 1, parameter=2.)
        # Create a parameterized rotation on qubit 1 , with an undefined angle, that will be variational
        RZ_gate = Gate("RZ", 1, parameter="an expression", is_variational=True)

        for gate in [H_gate, CNOT_gate, RX_gate, RZ_gate]:
            print(gate)

    def test_incorrect_gate(self):
        """ Test to catch a gate with inputs that do not make sense """

        self.assertRaises(ValueError, Gate, "H", -1)
        self.assertRaises(ValueError, Gate, "CNOT", 0, control=0.3)


if __name__ == "__main__":
    unittest.main()
