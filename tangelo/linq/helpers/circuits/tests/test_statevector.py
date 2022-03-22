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
import numpy as np

from tangelo.linq import Simulator
from tangelo.linq.helpers.circuits import StateVector
from tangelo.helpers.utils import installed_backends


class StateVectorTest(unittest.TestCase):

    def test_init(self):
        """Test initialization of the ansatz class."""
        n_qubits = 3
        v = np.full((2**n_qubits), 1.+1j)
        v /= np.linalg.norm(v)

        # Test raises ValueError for vector of length not equal to 2**(integer)
        self.assertRaises(ValueError, StateVector, v[0:7])
        # Test raises ValueError if order does is not "msq_first" or "lsq_first"
        self.assertRaises(ValueError, StateVector, v, "not_msq_first_or_lsq_first")

    def test_circuits_representations_lsq(self):
        """Test initializing and uncomputing circuits with cirq lsq_first order"""
        sim = Simulator("cirq")
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) + 1j*np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        v /= np.linalg.norm(v)

        sv = StateVector(v, order=sim.statevector_order)

        init_circ, phase = sv.initializing_circuit(return_phase=True)
        _, nsv = sim.simulate(init_circ, return_statevector=True)
        np.testing.assert_array_almost_equal(nsv*np.exp(1j*phase), v)

        uncomp_circ, phase = sv.uncomputing_circuit(return_phase=True)
        zero_state = np.zeros(8)
        zero_state[0] = 1
        _, nsv = sim.simulate(uncomp_circ, initial_statevector=v, return_statevector=True)
        np.testing.assert_array_almost_equal(nsv*np.exp(1j*phase), zero_state)

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_circuits_representations_msq(self):
        """Test initializing and uncomputing circuits with qulacs msq_first order"""
        sim = Simulator("qulacs")
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) + 1j*np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        v /= np.linalg.norm(v)

        sv = StateVector(v, order=sim.statevector_order)

        init_circ, phase = sv.initializing_circuit(return_phase=True)
        _, nsv = sim.simulate(init_circ, return_statevector=True)
        np.testing.assert_array_almost_equal(nsv*np.exp(1j*phase), v)

        uncomp_circ, phase = sv.uncomputing_circuit(return_phase=True)
        zero_state = np.zeros(8)
        zero_state[0] = 1
        _, nsv = sim.simulate(uncomp_circ, initial_statevector=v, return_statevector=True)
        np.testing.assert_array_almost_equal(nsv*np.exp(1j*phase), zero_state)


if __name__ == "__main__":
    unittest.main()
