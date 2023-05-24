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

from tangelo.linq import get_backend
from tangelo.helpers.utils import installed_backends
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.circuits.lcu import get_oaa_lcu_circuit, get_truncated_taylor_series

# Test for both "cirq" and if available "qulacs". These have different orderings.
# qiskit is not currently supported because does not have multi controlled general gates.
backends = ["cirq", "qulacs"] if "qulacs" in installed_backends else ["cirq"]
# Initiate Simulator using cirq for phase estimation tests as it has the same ordering as openfermion
# and we are using an exact eigenvector for testing.
sim_cirq = get_backend("cirq")


class LCUTest(unittest.TestCase):

    def test_get_truncated_taylor_series(self):
        """Test time-evolution of truncated Taylor series for different orders and times"""

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk", mol_H2_sto3g.n_active_sos, mol_H2_sto3g.n_active_electrons,
                                         True, 0)
        n_qubits_qu_op = math.ceil(math.log2(len(qu_op.terms)))
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
                taylor_circuit = get_truncated_taylor_series(qu_op, k, time)
                _, v = sim.simulate(sv_circuit + taylor_circuit, return_statevector=True)
                len_ancilla = 2**(k+k*n_qubits_qu_op+1)
                v = v.reshape([4, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, 4])[0, :]
                self.assertAlmostEqual(1, np.abs(v.conj().dot(exact)), delta=3.e-1**k)

        # Raise ValueError if Taylor series order is less than 1 or greater than 4
        # or imaginary coefficients in qubit operator
        self.assertRaises(ValueError, get_truncated_taylor_series, qu_op, 0, time)
        self.assertRaises(ValueError, get_truncated_taylor_series, qu_op * 1j, 2, time)

    def test_get_oaa_lcu_circuit(self):
        """Test time-evolution of truncated Taylor series for order k = 3 passing explicitly calculated
        qubit operator exponential"""

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk",
                                         mol_H2_sto3g.n_active_sos, mol_H2_sto3g.n_active_electrons, True, 0)
        time = 0.5
        # Generate explicit qubit operator exponential
        exp_qu_op = 1 + -1j*qu_op*time + (-1j*qu_op*time)**2/2 + (-1j*qu_op*time)**3/6
        exp_qu_op.compress()
        n_qubits_qu_op = math.ceil(math.log2(len(exp_qu_op.terms)))
        ham = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, vecs = np.linalg.eigh(ham)
        vec = (vecs[:, 0] + vecs[:, 1])/np.sqrt(2)

        exact = expm(-1j*ham*time)@vec
        len_ancilla = 2**(n_qubits_qu_op)

        for backend in backends:
            sim = get_backend(backend)
            statevector_order = sim.backend_info()["statevector_order"]
            sv = StateVector(vec, order=statevector_order)
            sv_circuit = sv.initializing_circuit()

            taylor_circuit = get_oaa_lcu_circuit(exp_qu_op)
            _, v = sim.simulate(sv_circuit + taylor_circuit, return_statevector=True)
            v = v.reshape([4, len_ancilla])[:, 0] if statevector_order == "lsq_first" else v.reshape([len_ancilla, 4])[0, :]
            self.assertAlmostEqual(1, np.abs(v.conj().dot(exact)), delta=1.e-3)

        # Test return of ValueError if 1-norm is greater than 2.
        self.assertRaises(ValueError, get_oaa_lcu_circuit, exp_qu_op+5)

    def test_controlled_time_evolution_by_phase_estimation_for_get_truncated_taylor_series(self):
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
        for i in range(13):
            wave_9 = np.kron(wave_9, np.array([1, 0]))

        qubit_list = [16, 15, 14]

        pe_circuit = get_qft_circuit(qubit_list)
        for i, qubit in enumerate(qubit_list):
            pe_circuit += get_truncated_taylor_series(qu_op, 3, t=-(2*np.pi)*2**i, control=qubit)
        pe_circuit += get_qft_circuit(qubit_list, inverse=True)

        freqs, _ = sim_cirq.simulate(pe_circuit, initial_statevector=wave_9)

        # Trace out all but final 3 indices
        trace_freq = dict()
        for key, value in freqs.items():
            trace_freq[key[-3:]] = trace_freq.get(key[-3:], 0) + value

        # State 9 has eigenvalue 0.25 so return should be 010 (0*1/2 + 1*1/4 + 0*1/8)
        self.assertAlmostEqual(trace_freq["010"], 1.0, delta=1.e-4)

    def test_controlled_time_evolution_by_phase_estimation_for_get_oaa_lcu_circuit(self):
        """ Verify that the controlled time-evolution is correct by calculating the eigenvalue of an eigenstate through
        phase estimation.
        """

        # Generate qubit operator with state 9 having eigenvalue 0.25
        qu_op = (QubitOperator("X0 X1", 0.125) + QubitOperator("Y1 Y2", 0.125) + QubitOperator("Z2 Z3", 0.125)
                 + QubitOperator("", 0.125))

        ham_mat = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, wavefunction = np.linalg.eigh(ham_mat)

        # break time into 6 parts so 1-norm is less than 2. i.e. can use Oblivious Amplitude Amplification
        time = -2 * np.pi / 6

        # Generate explicit qubit operator exponential
        exp_qu_op = 1 + -1j*qu_op*time + (-1j*qu_op*time)**2/2 + (-1j*qu_op*time)**3/6

        num_terms = len(exp_qu_op.terms)
        n_extra_qubits = math.ceil(np.log2(num_terms))

        # Kronecker product n_extra_qubits + 3 qubits in the zero state to eigenvector 9
        wave_9 = wavefunction[:, 9]
        for i in range(n_extra_qubits+3):
            wave_9 = np.kron(wave_9, np.array([1, 0]))

        # Phase estimation register in reversed order
        qubit_list = list(reversed(range(4 + n_extra_qubits, 4 + n_extra_qubits + 3)))

        # Build phase estimation circuit and simulate
        pe_circuit = get_qft_circuit(qubit_list)
        for i, qubit in enumerate(qubit_list):
            pe_circuit += (6 * 2**i) * get_oaa_lcu_circuit(exp_qu_op, control=qubit)
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
