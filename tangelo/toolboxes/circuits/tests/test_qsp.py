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
from scipy.linalg import expm
from openfermion import get_sparse_operator

from tangelo.linq import get_backend, backend_info
from tangelo.helpers.utils import installed_backends
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.circuits.lcu import get_lcu_qubit_op_info
from tangelo.toolboxes.circuits.qsp import get_qsp_hamiltonian_simulation_circuit, get_qsp_hamiltonian_simulation_qubit_list

# Test for both "cirq" and if available "qulacs". These have different orderings.
# qiskit is not currently supported because does not have multi controlled general gates.
backends = ["cirq", "qulacs"] if "qulacs" in installed_backends else ["cirq"]
# Initiate Simulator using cirq for phase estimation tests as it has the same ordering as openfermion
# and we are using an exact eigenvector for testing.
sim_cirq = get_backend("cirq")


class QSPTest(unittest.TestCase):

    def test_get_qsp_circuit(self):
        """Test QSP time-evolution"""

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk",
                                         mol_H2_sto3g.n_active_sos, mol_H2_sto3g.n_active_electrons, True, 0)
        # need to ensure eigenvalues are between -1 and 1
        qu_op /= 1.2
        ham = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, vecs = np.linalg.eigh(ham)
        vec = (vecs[:, 0] + vecs[:, 2])/np.sqrt(2)

        time = 1.9
        exact = expm(-1j*ham*time)@vec

        for backend in backends:
            sim = get_backend(backend)
            statevector_order = backend_info[backend]["statevector_order"]
            sv = StateVector(vec, order=statevector_order)
            sv_circuit = sv.initializing_circuit()

            # Tested for up to k = 5 but 5 is slow due to needing 23 qubits to simulate.
            qsp_circuit = get_qsp_hamiltonian_simulation_circuit(qu_op, time)
            _, v = sim.simulate(sv_circuit + qsp_circuit, return_statevector=True)
            _, m_qs, _ = get_lcu_qubit_op_info(qu_op)
            len_ancilla = 2**(len(m_qs) + 3)
            v = v.reshape([4, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, 4])[0, :]

            self.assertAlmostEqual(1, np.abs(v.conj().dot(exact)), delta=3.e-1)

    def test_controlled_time_evolution_by_phase_estimation_for_get_qsp_circuit(self):
        """ Verify that the controlled QSP time-evolution is correct by calculating the eigenvalue of an eigenstate through
        phase estimation.
        """

        # Generate qubit operator with state 9 having eigenvalue 0.25
        qu_op = (QubitOperator("X0 X1", 0.125) + QubitOperator("Y1 Y2", 0.125) + QubitOperator("Z2 Z3", 0.125)
                 + QubitOperator("", 0.125))

        ham_mat = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, wavefunction = np.linalg.eigh(ham_mat)

        # Kronecker product 13 qubits in the zero state to eigenvector 9 to account for ancilla qubits
        wave_9 = wavefunction[:, 9]
        for i in range(8):
            wave_9 = np.kron(wave_9, np.array([1, 0]))

        # Get QSP Hamiltonian simulation qubit_list
        qubit_list = get_qsp_hamiltonian_simulation_qubit_list(qu_op)
        self.assertEqual(qubit_list, list(range(9)))

        circuit_width = len(qubit_list)
        qubit_list = list(reversed(range(circuit_width, circuit_width+3)))

        pe_circuit = get_qft_circuit(qubit_list)
        for i, qubit in enumerate(qubit_list):
            pe_circuit += get_qsp_hamiltonian_simulation_circuit(qu_op, tau=-(2*np.pi)*2**i, control=qubit, eps=1.e-2)
        pe_circuit += get_qft_circuit(qubit_list, inverse=True)

        freqs, _ = sim_cirq.simulate(pe_circuit, initial_statevector=wave_9)

        # Trace out all but final 3 indices
        trace_freq = dict()
        for key, value in freqs.items():
            trace_freq[key[-3:]] = trace_freq.get(key[-3:], 0) + value

        # State 9 has eigenvalue 0.25 so return should be 010 (0*1/2 + 1*1/4 + 0*1/8)
        self.assertAlmostEqual(trace_freq["010"], 1.0, delta=1.e-4)


if __name__ == "__main__":
    unittest.main()
