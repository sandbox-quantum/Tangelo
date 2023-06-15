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

from tangelo.linq import get_backend
from tangelo.helpers.utils import installed_backends
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.circuits.multiproduct import get_multi_product_circuit, get_ajs_kjs

# Test for both "cirq" and if available "qulacs". These have different orderings.
# qiskit is not currently supported because does not have multi controlled general gates.
backends = ["cirq", "qulacs"] if "qulacs" in installed_backends else ["cirq"]
# Initiate Simulator using cirq for phase estimation tests as it has the same ordering as openfermion
# and we are using an exact eigenvector for testing.
sim_cirq = get_backend("cirq")


class MultiProductTest(unittest.TestCase):

    def test_time_evolution(self):
        """Test time-evolution of multi-product circuit for different orders"""

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk",
                                         mol_H2_sto3g.n_active_sos, mol_H2_sto3g.n_active_electrons,
                                         True, 0)

        ham = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, vecs = np.linalg.eigh(ham)
        vec = (vecs[:, 0] + vecs[:, 1])/np.sqrt(2)

        time = 1.9
        exact = expm(-1j*ham*time)@vec

        for backend in backends:
            sim = get_backend(backend)
            statevector_order = sim.backend_info()["statevector_order"]
            sv = StateVector(vec, order=statevector_order)
            sv_circuit = sv.initializing_circuit()

            # Tested for up to k = 5 but 5 is slow due to needing 23 qubits to simulate.
            for k in [1, 2, 3, 4]:
                taylor_circuit = get_multi_product_circuit(time, order=k, n_state_qus=2, operator=qu_op)
                _, v = sim.simulate(sv_circuit + taylor_circuit, return_statevector=True)
                _, _, n_ancilla = get_ajs_kjs(k)
                len_ancilla = 2**n_ancilla
                v = v.reshape([4, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, 4])[0, :]
                self.assertAlmostEqual(1, np.abs(v.conj().dot(exact)), delta=3.e-1**k)

        # Raise ValueError if order is less than 1 or greater than 6 or imaginary coefficients in qubit operator
        self.assertRaises(ValueError, get_multi_product_circuit, time, 0, 2, qu_op)
        self.assertRaises(ValueError, get_multi_product_circuit, time, 7, 2, qu_op)
        # Raise TypeError if order not integer
        self.assertRaises(TypeError, get_multi_product_circuit, time, 3., 2, qu_op)

    def test_controlled_time_evolution_by_phase_estimation(self):
        """ Verify that the controlled time-evolution is correct by calculating the eigenvalue of an eigenstate through
        phase estimation.
        """

        # Generate qubit operator with state 9 having eigenvalue 0.25
        qu_op = (QubitOperator("X0 X1", 0.125) + QubitOperator("Y1 Y2", 0.125) + QubitOperator("Z2 Z3", 0.125)
                 + QubitOperator("", 0.125))

        ham_mat = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, wavefunction = np.linalg.eigh(ham_mat)

        # Kronecker product 13 qubits in the zero state to eigenvector 9 to account for ancilla qubits
        wave_9 = wavefunction[:, 9]
        for i in range(6):
            wave_9 = np.kron(wave_9, np.array([1, 0]))

        qubit_list = [9, 8, 7]

        pe_circuit = get_qft_circuit(qubit_list)
        for i, qubit in enumerate(qubit_list):
            pe_circuit += get_multi_product_circuit(operator=qu_op, n_state_qus=4, order=5,
                                                    time=-(2*np.pi)*2**i, control=qubit)
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
