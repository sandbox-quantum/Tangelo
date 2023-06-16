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

import unittest

import numpy as np

from tangelo.linq import get_backend
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.helpers.utils import installed_backends
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.toolboxes.post_processing.histogram import Histogram
from tangelo.toolboxes.circuits.grid_circuits import get_psquared_circuit, get_xsquared_circuit

# Test for both "cirq" and if available "qulacs". These have different orderings.
# qiskit is not currently supported because does not have multi controlled general gates.
backends = ["cirq", "qulacs"] if "qulacs" in installed_backends else ["cirq"]
# Initiate Simulator using cirq for phase estimation tests as it has the same ordering as openfermion
# and we are using an exact eigenvector for testing.
sim_cirq = get_backend("cirq")


class GridTest(unittest.TestCase):

    def test_controlled_time_evolution_by_phase_estimation(self):
        """ Verify that the controlled time-evolution is correct by calculating the eigenvalue of an eigenstate of the
        harmonic oscillator (1/2/mass p^2/2 + 1/4*x^2 - 1/4) with mass=2 and ground state eigenvalue 1/8
        """

        n_qubits = 6
        n_pts = 2**n_qubits
        dx = 0.2
        x0 = dx*(n_pts//2 - 1/2)
        gridpts = np.linspace(-x0, x0, n_pts)
        mass = 2

        # Kronecker product 13 qubits in the zero state to eigenvector 9 to account for ancilla qubits
        wave_0 = np.exp(-1/2*(gridpts)**2)*np.sqrt(dx)/np.pi**(1/4)

        fft_list = [8, 7, 6]
        for backend in backends:
            sim = get_backend(backend)
            sim_order = sim.backend_info()["statevector_order"]

            qubit_list = list(reversed(range(n_qubits))) if sim_order == "lsq_first" else list((range(n_qubits)))
            start_circ = StateVector(wave_0, order=sim_order).initializing_circuit()

            pe_circuit = start_circ + get_qft_circuit(fft_list)
            for i, qubit in enumerate(fft_list):
                xsquared = get_xsquared_circuit(-2*np.pi/20, dx, 1/4, x0, -1/8., qubit_list=qubit_list, control=qubit)
                psquared = get_psquared_circuit(-2*np.pi/10, dx, mass, qubit_list=qubit_list, control=qubit)
                pe_circuit += (xsquared + psquared + xsquared) * (10 * 2**i)
            pe_circuit += get_qft_circuit(fft_list, inverse=True)

            freqs, _ = sim.simulate(pe_circuit)

            # Trace out all but final 3 indices
            hist = Histogram(freqs)
            hist.remove_qubit_indices(*qubit_list)
            trace_freq = hist.frequencies

        # State 0 has eigenvalue 0.125 so return should be 001 (0*1/2 + 0*1/4 + 1*1/8)
        self.assertAlmostEqual(trace_freq["001"], 1.0, delta=1.e-3)


if __name__ == "__main__":
    unittest.main()
