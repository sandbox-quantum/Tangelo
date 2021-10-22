import unittest
from cirq.sim import state_vector

from scipy.linalg import expm
import numpy as np
from numpy.linalg import eigh
from openfermion import get_sparse_operator

from qsdk.backendbuddy import Simulator, Circuit, Gate
from qsdk.toolboxes.operators import FermionOperator, QubitOperator
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.molecule_library import mol_H4_sto3g
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import trotterize, qft_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import derangement_circuit, controlled_pauliwords

# Initiate simulator, use cirq as it uses lsq_first like openfermion
sims = [Simulator(target="cirq"), Simulator(target="qulacs"), Simulator(target="qiskit")]

fermion_operator = mol_H4_sto3g._get_fermionic_hamiltonian()


def assert_freq_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two frequency dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Dictionary entries beyond tolerance {atol}: \n{d1} \n{d2}")
    return True


class ansatz_utils_Test(unittest.TestCase):

    def test_trotterize_fermion_input(self):
        """ Verify that the time evolution is correct for different mappings and a fermionic
            hamiltonian input
        """

        time = 0.2
        for mapping in ['jw', 'bk', 'scbk']:
            reference_circuit = get_reference_circuit(n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                      n_electrons=mol_H4_sto3g.n_active_electrons,
                                                      mapping=mapping,
                                                      up_then_down=True)
            _, refwave = sims[0].simulate(reference_circuit, return_statevector=True)

            qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=fermion_operator,
                                                         mapping=mapping,
                                                         n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                         n_electrons=mol_H4_sto3g.n_active_electrons,
                                                         up_then_down=True)

            ham_mat = get_sparse_operator(qubit_hamiltonian).toarray()

            evolve_exact = expm(-1j * time * ham_mat) @ refwave

            options = {"up_then_down": True,
                       "qubit_mapping": mapping,
                       'n_spinorbitals': mol_H4_sto3g.n_active_sos,
                       'n_electrons': mol_H4_sto3g.n_active_electrons}
            tcircuit, phase = trotterize(fermion_operator,
                                         trotter_order=1,
                                         num_trotter_steps=1,
                                         time=time,
                                         mapping_options=options)
            _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
            wavefunc *= phase
            overlap = np.dot(np.conj(evolve_exact), wavefunc)
            self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

    def test_trotterize_qubit_input(self):
        """ Verify that the time evolution is correct for different mappings and a qubit_hamiltonian input"""

        time = 0.2
        for mapping in ['jw', 'bk', 'scbk']:
            reference_circuit = get_reference_circuit(n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                      n_electrons=mol_H4_sto3g.n_active_electrons,
                                                      mapping=mapping,
                                                      up_then_down=True)
            _, refwave = sims[0].simulate(reference_circuit, return_statevector=True)

            qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=fermion_operator,
                                                         mapping=mapping,
                                                         n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                         n_electrons=mol_H4_sto3g.n_active_electrons,
                                                         up_then_down=True)

            ham_mat = get_sparse_operator(qubit_hamiltonian).toarray()

            evolve_exact = expm(-1j * time * ham_mat) @ refwave

            tcircuit, phase = trotterize(qubit_hamiltonian,
                                         trotter_order=1,
                                         num_trotter_steps=1,
                                         time=time)
            _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
            wavefunc *= phase
            overlap = np.dot(np.conj(evolve_exact), wavefunc)
            self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

    def test_trotterize_different_order_and_steps(self):
        """ Verify that the time evolution is correct for different orders and number of steps
            with a qubit_hamiltonian input"""

        time = 0.2
        mapping = 'bk'
        reference_circuit = get_reference_circuit(n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                  n_electrons=mol_H4_sto3g.n_active_electrons,
                                                  mapping=mapping,
                                                  up_then_down=True)
        _, refwave = sims[0].simulate(reference_circuit, return_statevector=True)

        qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=fermion_operator,
                                                     mapping=mapping,
                                                     n_spinorbitals=mol_H4_sto3g.n_active_sos,
                                                     n_electrons=mol_H4_sto3g.n_active_electrons,
                                                     up_then_down=True)

        ham_mat = get_sparse_operator(qubit_hamiltonian).toarray()

        evolve_exact = expm(-1j * time * ham_mat) @ refwave

        for trotter_order in [1, 2]:

            tcircuit, phase = trotterize(qubit_hamiltonian,
                                         trotter_order=trotter_order,
                                         num_trotter_steps=1,
                                         time=time)
            _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
            wavefunc *= phase
            overlap = np.dot(np.conj(evolve_exact), wavefunc)
            self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

        for num_trotter_steps in [1, 2]:

            tcircuit, phase = trotterize(qubit_hamiltonian,
                                         trotter_order=1,
                                         num_trotter_steps=num_trotter_steps,
                                         time=time)
            _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
            wavefunc *= phase
            overlap = np.dot(np.conj(evolve_exact), wavefunc)
            self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

    def test_trotterize_fermionic_input_different_times(self):
        """ Verify that the time evolution is correct for a FermionOperator input with different times
            for each term
        """

        mapping = 'jw'
        # generate random Hermitian FermionOperator
        fermion_operator_list = [FermionOperator('0^ 3', 0.5) + FermionOperator('3^ 0', 0.5)]
        fermion_operator_list.append(FermionOperator('1^ 2', 0.5) + FermionOperator('2^ 1', 0.5))
        fermion_operator_list.append(FermionOperator('1^ 3', 0.5) + FermionOperator('3^ 1', 0.5))

        # time is twice as long as each Hermitian Operator has two terms
        time = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]

        # Build referenc circuit and obtain reference wavefunction
        reference_circuit = Circuit([Gate('X', 0), Gate('X', 3)], n_qubits=4)
        _, refwave = sims[0].simulate(reference_circuit, return_statevector=True)

        evolve_exact = refwave
        total_fermion_operator = FermionOperator()
        # evolve each term separately and apply to resulting wavefunction
        for i in range(3):
            total_fermion_operator += fermion_operator_list[i]
            qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=fermion_operator_list[i],
                                                         mapping=mapping)
            ham_mat = get_sparse_operator(qubit_hamiltonian, n_qubits=4).toarray()

            evolve_exact = expm(-1j * time[2*i] * ham_mat) @ evolve_exact

        # Apply trotter-suzuki steps using different times for each term
        tcircuit, phase = trotterize(total_fermion_operator,
                                     trotter_order=1,
                                     num_trotter_steps=1,
                                     time=time)
        _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
        wavefunc *= phase

        overlap = np.dot(np.conj(evolve_exact), wavefunc)
        self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

    def test_trotterize_qubit_input_different_times(self):
        """ Verify that the time evolution is correct for a QubitOperator input with different times
            for each term
        """

        # Generate random qubit operator
        qubit_operator_list = [QubitOperator('X0 Y1', 0.5)]
        qubit_operator_list.append(QubitOperator('Y1 Z2', 0.5))
        qubit_operator_list.append(QubitOperator('Y2 X3', 0.5))

        time = [0.1, 0.2, 0.3]

        # Generate initial wavefunction
        reference_circuit = Circuit([Gate('X', 0), Gate('X', 3)], n_qubits=4)
        _, refwave = sims[0].simulate(reference_circuit, return_statevector=True)

        # Exactly evolve for each time step
        evolve_exact = refwave
        total_qubit_operator = QubitOperator()
        for i in range(3):
            total_qubit_operator += qubit_operator_list[i]
            ham_mat = get_sparse_operator(qubit_operator_list[i], n_qubits=4).toarray()

            evolve_exact = expm(-1j * time[i] * ham_mat) @ evolve_exact

        # Apply trotter-suzuki with different times for each qubit operator term
        tcircuit, phase = trotterize(total_qubit_operator,
                                     trotter_order=1,
                                     num_trotter_steps=1,
                                     time=time)
        _, wavefunc = sims[0].simulate(tcircuit, return_statevector=True, initial_statevector=refwave)
        wavefunc *= phase

        overlap = np.dot(np.conj(evolve_exact), wavefunc)
        self.assertAlmostEqual(overlap, 1.0, delta=1e-3)

    def test_qft_by_phase_estimation(self):
        n_qubits = 4
        qubit_list = [2, 1, 0]
        # Generate state with eigenvalue -1 of X operator exp(2*pi*i*phi) phi=1/2
        gate_list = [Gate('X', target=n_qubits-1), Gate('H', target=n_qubits-1)]
        pe_circuit = Circuit(gate_list, n_qubits=n_qubits)
        qft = qft_circuit(qubit_list, n_qubits_in_circuit=n_qubits)
        pe_circuit += qft
        controlled_unitaries = []
        for i, qubit in enumerate(qubit_list):
            for j in range(2**i):
                controlled_unitaries += [Gate('CNOT', target=n_qubits-1, control=qubit)]
        pe_circuit += Circuit(controlled_unitaries, n_qubits=n_qubits)
        iqft = qft_circuit(qubit_list, n_qubits_in_circuit=n_qubits, inverse=True)
        pe_circuit += iqft
        for sim in sims:
            freqs, _ = sim.simulate(pe_circuit)
            target_freq_dict = {'1000': 0.5, '1001': 0.5}  # 1 * 1/2 + 0 * 1/4 + 0 * 1/8
            assert_freq_dict_almost_equal(target_freq_dict, freqs, atol=1.e-7)

    def test_controlled_time_evolution_by_phase_estimation(self):
        """ Verify that the time evolution is correct for a QubitOperator input with different times
            for each term
        """

        # Generate random qubit operator
        qu_op = QubitOperator('X0 X1', 0.125) + QubitOperator('Y1 Y2', 0.125) + QubitOperator('Z2 Z3', 0.125)
        qu_op += QubitOperator('', 0.125)

        ham_mat = get_sparse_operator(qu_op).toarray()
        _, wavefunction = eigh(ham_mat)

        # State 9 has eigenvalue 0.25 so return should be 0100 (0*1/2 + 1*1/4 + 0*1/8 + 0*1/16)
        ground_wave = wavefunction[:, 9]
        ground_wave = np.kron(ground_wave, np.array([1, 0]))
        ground_wave = np.kron(ground_wave, np.array([1, 0]))
        ground_wave = np.kron(ground_wave, np.array([1, 0]))
        ground_wave = np.kron(ground_wave, np.array([1, 0]))

        n_qubits = 8

        qubit_list = [7, 6, 5, 4]
        gate_list = []
        pe_circuit = Circuit(gate_list, n_qubits=n_qubits)
        qft = qft_circuit(qubit_list, n_qubits_in_circuit=n_qubits)
        pe_circuit += qft
        for i, qubit in enumerate(qubit_list):
            u_circuit = trotterize(qu_op,
                                   trotter_order=1,
                                   num_trotter_steps=10,
                                   time=-2*np.pi,
                                   control=qubit)
            for j in range(2**i):
                pe_circuit += u_circuit[0]
        iqft = qft_circuit(qubit_list, n_qubits_in_circuit=n_qubits, inverse=True)
        pe_circuit += iqft
        for sim in sims:
            freqs, _ = sim.simulate(pe_circuit, initial_statevector=ground_wave)
            trace_freq = dict()
            for key, value in freqs.items():
                if key[-4:] in trace_freq:
                    trace_freq[key[-4:]] += value
                else:
                    trace_freq[key[-4:]] = value
            self.assertAlmostEqual(trace_freq['0100'], 1.0, delta=2)

    def test_controlled_swap(self):
        cswap = Circuit([Gate('CSWAP', target=[1, 2], control=0)], n_qubits=3)

        # initialize in '110', returns '101'
        init_gates = [Gate('X', target=0), Gate('X', target=1)]
        circuit = Circuit(init_gates, n_qubits=3) + cswap
        for sim in sims:
            freqs, _ = sim.simulate(circuit, return_statevector=True)
            assert_freq_dict_almost_equal({'101': 1.0}, freqs, atol=1.e-7)

        # initialize in '010' returns '010'
        init_gates = [Gate('X', target=1)]
        circuit = Circuit(init_gates, n_qubits=3) + cswap
        for sim in sims:
            freqs, _ = sim.simulate(circuit, return_statevector=True)
            assert_freq_dict_almost_equal({'010': 1.0}, freqs, atol=1.e-7)

    def test_derangement_circuit_by_estimating_pauli_string(self):
        """ Verify that tr(rho^3 pa) for a pauliword pa is correct.
        Uses the exponential error suppression circuit
        """

        # Generate random qubit operator
        qu_op = QubitOperator('X0 Y1', 0.125) + QubitOperator('Y1 Y2', 0.125) + QubitOperator('Z2 Z3', 0.125)
        pa = QubitOperator('X0 X1 X2 X3', 1)

        ham_mat = get_sparse_operator(qu_op).toarray()
        _, wavefunction = eigh(ham_mat)
        pamat = get_sparse_operator(pa).toarray()

        # State 9 has eigenvalue 0.25 so return should be 0100 (0*1/2 + 1*1/4 + 0*1/8 + 0*1/16)
        mixed_wave = np.sqrt(3)/2*wavefunction[:, 9] + 1/2*wavefunction[:, 0]
        mixed_wave_3 = np.kron(np.kron(mixed_wave, mixed_wave), mixed_wave)
        full_start_vec = np.kron(mixed_wave_3, np.array([1, 0]))
        rho = np.outer(mixed_wave, mixed_wave)
        rho3 = rho @ rho @ rho
        exact = np.trace(rho3 @ pamat)

        n_qubits = 13

        qubit_list = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        rho3_pa_circuit = Circuit([Gate('H', target=12)], n_qubits=n_qubits)
        derange_circuit = derangement_circuit(qubit_list, control=12, n_qubits=n_qubits)
        cpas = controlled_pauliwords(pa, control=12, n_qubits=n_qubits)
        rho3_pa_circuit += derange_circuit + cpas[0]
        rho3_pa_circuit += Circuit([Gate('H', target=12)], n_qubits=n_qubits)

        exp_op = QubitOperator('Z12', 1)
        measured = sims[0].get_expectation_value(exp_op, rho3_pa_circuit, initial_statevector=full_start_vec)
        self.assertAlmostEqual(measured, exact, places=6)


if __name__ == "__main__":
    unittest.main()
