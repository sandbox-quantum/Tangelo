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

"""A test class to check that the simulator class functionalities are behaving
as expected for the symbolic backend.
"""

import unittest
from math import pi

from tangelo.helpers.utils import assert_freq_dict_almost_equal
from tangelo.linq import Gate, Circuit
from tangelo.helpers.utils import installed_backends
from tangelo.linq.target.target_sympy import SympySimulator


class TestSymbolicSimulate(unittest.TestCase):

    @unittest.skipIf("sympy" not in installed_backends, "Test Skipped: Sympy backend not available \n")
    def test_simple_simulate(self):
        """Test simulate of a simple rotation gate with a symbolic parameter."""

        from sympy import symbols, cos, sin

        simple_circuit = Circuit([Gate("RY", 0, parameter="alpha")])
        backend = SympySimulator()
        probs, _ = backend.simulate(simple_circuit, return_statevector=False)

        alpha = symbols("alpha", real=True)

        self.assertDictEqual(probs, {"0": (cos(alpha/2))**2, "1": (sin(alpha/2))**2})

    @unittest.skipIf("sympy" not in installed_backends, "Test Skipped: Sympy backend not available \n")
    def test_simulate_with_control(self):
        """Test simulate of a control rotation gate with a symbolic parameter."""

        from sympy import symbols, cos, sin

        backend = SympySimulator()

        no_action_circuit = Circuit([Gate("CRY", 1, 0, parameter="alpha")])
        no_action_probs, _ = backend.simulate(no_action_circuit, return_statevector=False)

        self.assertDictEqual(no_action_probs, {"00": 1.})

        action_circuit = Circuit([Gate("X", 0), Gate("CRY", 1, 0, parameter="alpha")])
        action_probs, _ = backend.simulate(action_circuit, return_statevector=False)
        alpha = symbols("alpha", real=True)

        self.assertDictEqual(action_probs, {"10": (cos(alpha/2))**2, "11": (sin(alpha/2))**2})

    @unittest.skipIf("sympy" not in installed_backends, "Test Skipped: Sympy backend not available \n")
    def test_evaluate_bell_state(self):
        """Test the numerical evaluation to a known state (Bell state)."""

        backend = SympySimulator()

        variable_bell_circuit = Circuit([Gate("RY", 0, parameter="alpha"), Gate("CNOT", 1, 0)])
        variable_bell_probs, _ = backend.simulate(variable_bell_circuit, return_statevector=False)

        # Replace alpha by pi/2.
        numerical_bell_probs = {
            bitstring: prob.subs(list(prob.free_symbols)[0], pi/2) for
            bitstring, prob in variable_bell_probs.items()
        }

        assert_freq_dict_almost_equal(numerical_bell_probs, {"00": 0.5, "11": 0.5}, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
