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
import math

import numpy as np
from scipy.linalg import expm
from openfermion import get_sparse_operator

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.helpers.utils import installed_backends
from tangelo.linq import get_backend, Circuit
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize
from tangelo.toolboxes.circuits.discrete_clock import get_discrete_clock_circuit
from tangelo.toolboxes.circuits.grid_circuits import get_psquared_circuit, get_xsquared_circuit

# Test for both "cirq" and if available "qulacs". These have different orderings.
# qiskit is not currently supported because does not have multi controlled general gates.
backends = ["cirq", "qulacs"] if "qulacs" in installed_backends else ["cirq"]
# Initiate Simulator using cirq for phase estimation tests as it has the same ordering as openfermion
# and we are using an exact eigenvector for testing.
sim_cirq = get_backend("cirq")


class DiscreteClockTest(unittest.TestCase):

    def test_time_independant_hamiltonian(self):
        """Test time-evolution of discrete clock for a time-independant Hamiltonian"""

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk", mol_H2_sto3g.n_active_sos, mol_H2_sto3g.n_active_electrons,
                                         True, 0)

        ham = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, vecs = np.linalg.eigh(ham)
        vec = (vecs[:, 0] + vecs[:, 1])/np.sqrt(2)

        time = 10.
        exact = expm(-1j*ham*time)@vec

        def trotter_func(t0, time, n_trotter_steps, control):
            return trotterize(operator=qu_op, time=time, n_trotter_steps=n_trotter_steps, control=control, trotter_order=2)

        for backend in backends:
            sim = get_backend(backend)
            statevector_order = sim.backend_info()["statevector_order"]
            sv = StateVector(vec, order=statevector_order)
            sv_circuit = sv.initializing_circuit()

            for k in [2, 3]:
                taylor_circuit = get_discrete_clock_circuit(trotter_func=trotter_func, trotter_kwargs={},
                                                            time=time, mp_order=k, n_state_qus=2, n_time_steps=4)
                _, v = sim.simulate(sv_circuit + taylor_circuit, return_statevector=True)
                n_ancilla = 2 + math.ceil(np.log2(k+2))
                len_ancilla = 2**n_ancilla
                v = v.reshape([4, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, 4])[0, :]
                self.assertAlmostEqual(1, np.abs(v.conj().dot(exact)), delta=1.e-1**k)

    def test_time_dependant_hamiltonian(self):
        """Test time-evolution of discrete clock for a time-dependant Hamiltonian taken from
        arXiv: 1412.1802 H = 1/2/m * p^2 + (4*exp(-2*t) - 1/16) * x^2 - 2*exp(-t) with mass=1/2
        and exact answer (2/pi)^(1/4)*exp(-x^2*exp(-t) - 1/4*t + 1j/8*x^2)"""

        n_qubits = 6
        n_pts = 2**n_qubits
        dx = 0.2
        x0 = dx*(n_pts//2 - 1/2)
        gridpts = np.linspace(-x0, x0, n_pts)
        mass = 1/2
        time = 1.

        def psiexact(xpts, t):
            return (2/np.pi)**(1/4)*np.exp(-xpts**2*np.exp(-t)-1/4*t+1j/8*xpts**2)*np.sqrt(dx)

        exact = psiexact(gridpts, time)

        def trotter_func(t0, time, n_trotter_steps, control, dx, qubit_list):
            circ = Circuit()
            dt = time/n_trotter_steps
            p2 = get_psquared_circuit(dt/2, dx, mass, qubit_list, control)
            for i in range(n_trotter_steps):
                th = t0 + (i + 1/2) * dt
                circ += p2
                circ += get_xsquared_circuit(dt, dx, (4*np.exp(-2*th) - 1/16), x0, -2*np.exp(-th), qubit_list, control)
                circ += p2
            return circ

        for backend in backends:
            sim = get_backend(backend)
            statevector_order = sim.backend_info()["statevector_order"]
            vec = psiexact(gridpts, 0)
            sv = StateVector(vec, order=statevector_order)
            sv_circuit, phase = sv.initializing_circuit(return_phase=True)

            for k in [2, 3]:
                qubit_list = list(reversed(range(n_qubits))) if statevector_order == "lsq_first" else list((range(n_qubits)))
                taylor_circuit = get_discrete_clock_circuit(trotter_func=trotter_func,
                                                            trotter_kwargs={"dx": dx, "qubit_list": qubit_list}, time=time,
                                                            mp_order=k, n_state_qus=6, n_time_steps=2)
                _, v = sim.simulate(sv_circuit + taylor_circuit, return_statevector=True)
                n_ancilla = 1 + math.ceil(np.log2(k+2))
                len_ancilla = 2**n_ancilla
                v = v.reshape([n_pts, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, n_pts])[0, :]
                self.assertAlmostEqual(1, (v.conj().dot(exact)*np.exp(-1j*phase)).real, delta=1.e-1**k)


if __name__ == "__main__":
    unittest.main()
