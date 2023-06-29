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

"""
    A test class to check that the features related to noisy simulation are working as expected.
"""

import unittest

import numpy as np

from tangelo.linq import Gate, Circuit, get_backend, backend_info
from tangelo.linq.noisy_simulation import NoiseModel, get_qiskit_noise_dict
from tangelo.helpers.utils import default_simulator, installed_backends, assert_freq_dict_almost_equal
from tangelo.toolboxes.operators import QubitOperator

# Noisy simulation: circuits, noise models, references
cn1 = Circuit([Gate('X', target=0)])
cn2 = Circuit([Gate('CNOT', target=1, control=0)])
circuit_mixed = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 1, parameter=-1.), Gate("MEASURE", 0), Gate("X", 0)])

nmp, nmd, nmc, nmm = NoiseModel(), NoiseModel(), NoiseModel(), NoiseModel()
# nmp: pauli noise with equal probabilities, on X and CNOT gates
nmp.add_quantum_error("X", 'pauli', [1 / 3] * 3)
nmp.add_quantum_error("CNOT", 'pauli', [1 / 3] * 3)
# nmd: depol noise with prob 1. on X and CNOT gates
nmd.add_quantum_error("X", 'depol', 1.)
nmd.add_quantum_error("CNOT", 'depol', 1.)
# nmc: cumulates 2 Pauli noises (here, is equivalent to no noise, as it applies Y twice when X is ran)
nmc.add_quantum_error("X", 'pauli', [0., 1., 0.])
nmc.add_quantum_error("X", 'depol', 4/3)
# nmm: only apply noise to X gate
nmm.add_quantum_error("X", 'pauli', [0.2, 0., 0.])

ref_pauli1 = {'1': 1 / 3, '0': 2 / 3}
ref_pauli2 = {'01': 2 / 9, '11': 4 / 9, '10': 2 / 9, '00': 1 / 9}
ref_depol1 = {'1': 1 / 2, '0': 1 / 2}
ref_depol2 = {'01': 1 / 4, '11': 1 / 4, '10': 1 / 4, '00': 1 / 4}
ref_cumul = {'0': 1/3, '1': 2/3}
ref_mixed = {'10': 0.2876, '11': 0.0844, '01': 0.1472, '00': 0.4808}
ref_mixed_0 = {'00': 0.1488, '10': 0.6113, '01': 0.0448, '11': 0.1950}


class TestSimulate(unittest.TestCase):

    def test_noisy_simulation_not_supported(self):
        """
            Ensures that an error is returned if user attempts to run noisy simulation on a backend that does
            not support it as part of this package.
        """
        for b, s in backend_info.items():
            if not s['noisy_simulation']:
                self.assertRaises(ValueError, get_backend, target=b, n_shots=1, noise_model=True)

    def test_cannot_cumulate_same_noise_same_gate(self):
        """ Ensures an error is returned if user attempts to cumulate the same type of noise on the same gate """
        nm = NoiseModel()
        nm.add_quantum_error('X', 'depol', 0.2)
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'depol', 0.3)

    def test_unsupported_noise_channel(self):
        """ Ensures an error is returned if user attempts to user an unsupported noise channel """
        nm = NoiseModel()
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'incorrect_input', 0.3)

    def test_incorrect_arguments(self):
        """ Ensures an error is returned if noise parameters are incorrect """
        nm = NoiseModel()
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'pauli', 0.3)
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'depol', 1)
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'depol', [0.3, 0.2, 0.1])

    # TODO: replace the noise channel name by one supported by agnostic simulator, but not with qiskit backend later
    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_unsupported_noise_channel_qiskit(self):
        """ Ensures an error is returned if user attempts to user an unsupported noise channel, for qiskit """
        nm = NoiseModel()
        self.assertRaises(ValueError, nm.add_quantum_error, 'X', 'dummy', 0.3)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_qiskit_noise_dictionary_rotations(self):
        """ Generate noise dictionary using qiskit gates as keys. Map rotation gates to U-gates with no redundancy
         Ensure results as expected."""

        nt, np = 'pauli', [0.5, 0.25, 0.25]
        nm = NoiseModel()
        for g in {"RX", "RY", "RZ"}:
            nm.add_quantum_error(g, nt, np)

        qnd = get_qiskit_noise_dict(nm)
        for g in {'u1', 'u2', 'u3'}:
            assert(g in qnd)
            assert(qnd[g] == [(nt, np)])

    @unittest.skipIf("qulacs" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_noisy_simulation_qulacs(self):
        """
            Test noisy simulation through qulacs.
            Currently tested: pauli noise, depolarization noise (1 and 2 qubit gates)
        """

        # Pauli noise for one- and two-qubit gates. Circuits are only a X gate, or just a CNOT gate.
        s_nmp = get_backend(target='qulacs', n_shots=10**6, noise_model=nmp)
        res_pauli1, _ = s_nmp.simulate(cn1)
        assert_freq_dict_almost_equal(res_pauli1, ref_pauli1, 1e-2)
        res_pauli2, _ = s_nmp.simulate(cn2)
        assert_freq_dict_almost_equal(res_pauli2, ref_pauli2, 1e-2)

        # Depol noise for one- and two-qubit gates. Circuits are only a X gate or just a CNOT gate.
        s_nmd = get_backend(target='qulacs', n_shots=10**6, noise_model=nmd)
        res_depol1, _ = s_nmd.simulate(cn1)
        assert_freq_dict_almost_equal(res_depol1, ref_depol1, 1e-2)
        res_depol2, _ = s_nmd.simulate(cn2)
        assert_freq_dict_almost_equal(res_depol2, ref_depol2, 1e-2)

        # Cumulate several noises on a given gate (here noise simplifies to identity)
        s_nmc = get_backend(target='qulacs', n_shots=10**6, noise_model=nmc)
        res_cumul, _ = s_nmc.simulate(cn1)
        assert_freq_dict_almost_equal(res_cumul, ref_cumul, 1e-2)

        s_nmm = get_backend(target="qulacs", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed)
        assert_freq_dict_almost_equal(res_mixed, ref_mixed, 7.e-2)

        s_nmm = get_backend(target="qulacs", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed, desired_meas_result="0")
        assert_freq_dict_almost_equal(ref_mixed_0, res_mixed, 7.e-2)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_noisy_simulation_qiskit(self):
        """
            Test noisy simulation through qiskit.
            Currently tested: pauli noise, depolarization noise (1 and 2 qubit gates)
        """

        # Pauli noise for one- and two-qubit gates. Circuits are only a X gate, or just a CNOT gate.
        s_nmp = get_backend(target='qiskit', n_shots=10**6, noise_model=nmp)
        res_pauli1, _ = s_nmp.simulate(cn1)
        assert_freq_dict_almost_equal(res_pauli1, ref_pauli1, 1e-2)
        res_pauli2, _ = s_nmp.simulate(cn2)
        assert_freq_dict_almost_equal(res_pauli2, ref_pauli2, 1e-2)

        # Depol noise for one- and two-qubit gates. Circuits are only a X gate or just a CNOT gate.
        s_nmd = get_backend(target='qiskit', n_shots=10**6, noise_model=nmd)
        res_depol1, _ = s_nmd.simulate(cn1)
        assert_freq_dict_almost_equal(res_depol1, ref_depol1, 1e-2)
        res_depol2, _ = s_nmd.simulate(cn2)
        assert_freq_dict_almost_equal(res_depol2, ref_depol2, 1e-2)

        # Cumulate several noises on a given gate (here noise simplifies to identity)
        s_nmp = get_backend(target='qiskit', n_shots=10**6, noise_model=nmc)
        res_cumul, _ = s_nmp.simulate(cn1)
        assert_freq_dict_almost_equal(res_cumul, ref_cumul, 1e-2)

        s_nmm = get_backend(target="qiskit", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed)
        assert_freq_dict_almost_equal(ref_mixed, res_mixed, 7.e-2)

        # Test noise with desired measurement result
        s_nmm = get_backend(target="qiskit", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed, desired_meas_result="0")
        assert_freq_dict_almost_equal(ref_mixed_0, res_mixed, 7.e-2)

    @unittest.skipIf("cirq" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_noisy_simulation_cirq(self):
        """
            Test noisy simulation through cirq.
            Currently tested: pauli noise, depolarization noise (1 and 2 qubit gates)
        """

        # Pauli noise for one- and two-qubit gates. Circuits are only a X gate, or just a CNOT gate.
        s_nmp = get_backend(target='cirq', n_shots=10**6, noise_model=nmp)
        res_pauli1, _ = s_nmp.simulate(cn1)
        assert_freq_dict_almost_equal(res_pauli1, ref_pauli1, 1e-2)
        res_pauli2, _ = s_nmp.simulate(cn2)
        assert_freq_dict_almost_equal(res_pauli2, ref_pauli2, 1e-2)

        # Depol noise for one- and two-qubit gates. Circuits are only a X gate or just a CNOT gate.
        s_nmd = get_backend(target='cirq', n_shots=10**6, noise_model=nmd)
        res_depol1, _ = s_nmd.simulate(cn1)
        assert_freq_dict_almost_equal(res_depol1, ref_depol1, 1e-2)
        res_depol2, _ = s_nmd.simulate(cn2)
        assert_freq_dict_almost_equal(res_depol2, ref_depol2, 1e-2)

        # Cumulate several noises on a given gate (here noise simplifies to identity)
        s_nmc = get_backend(target='cirq', n_shots=10**6, noise_model=nmc)
        res_cumul, _ = s_nmc.simulate(cn1)
        assert_freq_dict_almost_equal(res_cumul, ref_cumul, 1e-2)

        # Noisy mixed state without returning mid-circuit measurements
        s_nmm = get_backend(target="cirq", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed)
        assert_freq_dict_almost_equal(ref_mixed, res_mixed, 7.e-2)

        s_nmm = get_backend(target="cirq", n_shots=10**4, noise_model=nmm)
        res_mixed, _ = s_nmm.simulate(circuit_mixed, desired_meas_result="0")
        assert_freq_dict_almost_equal(ref_mixed_0, res_mixed, 7.e-2)

        # Noisy mixed-state with specified measurement result and returning density matrix
        s_nmm = get_backend(target="cirq", n_shots=10**4, noise_model=nmm)
        res_mixed, sv = s_nmm.simulate(circuit_mixed, desired_meas_result="0", return_statevector=True)
        assert_freq_dict_almost_equal(ref_mixed_0, res_mixed, 7.e-2)
        exact_sv = np.array([[ 0.15403023 + 0.j, -0.08414710 - 0.j,  0.00000000 + 0.j,  0.00000000 - 0.j],
                             [-0.08414710 - 0.j,  0.04596977 + 0.j,  0.00000000 - 0.j,  0.00000000 + 0.j],
                             [ 0.00000000 + 0.j,  0.00000000 - 0.j,  0.61612092 + 0.j, -0.33658839 - 0.j],
                             [ 0.00000000 - 0.j,  0.00000000 + 0.j, -0.33658839 - 0.j,  0.18387908 + 0.j]])
        np.testing.assert_array_almost_equal(sv, exact_sv)

    @unittest.skipIf("stim" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_noisy_simulation_stim(self):
        """
            Test noisy simulation through stim.
            Currently tested: pauli noise, depolarization noise (1 and 2 qubit gates)
        """

        # Pauli noise for one- and two-qubit gates. Circuits are only a X gate, or just a CNOT gate.
        s_nmp = get_backend(target='stim', n_shots=10**6, noise_model=nmp)
        res_pauli1, _ = s_nmp.simulate(cn1)
        assert_freq_dict_almost_equal(res_pauli1, ref_pauli1, 1e-2)
        res_pauli2, _ = s_nmp.simulate(cn2)
        assert_freq_dict_almost_equal(res_pauli2, ref_pauli2, 1e-2)

    def test_get_expectation_value_noisy(self):
        """Test of the get_expectation_value function with a noisy simulator"""
        # Test Hamiltonian.
        H = QubitOperator()
        H.terms = {(): -14.41806525945003, ((0, 'Z'),): 0.0809953994342687,
                   ((1, 'Z'),): 0.0809953994342687, ((0, 'Z'), (1, 'Z')): 0.0077184273651725865,
                   ((0, 'X'), (1, 'X')): 0.0758664717894615}

        # Hard coding of test circuit.
        circuit = Circuit()
        circuit.add_gate(Gate("RX", 0, parameter=3.141595416808))
        circuit.add_gate(Gate("RX", 1,  parameter=3.141588753134))
        circuit.add_gate(Gate("H", 0))
        circuit.add_gate(Gate("RX", 1, parameter=1.570796326795))
        circuit.add_gate(Gate("CNOT", 1, 0))
        circuit.add_gate(Gate("RZ", 1, parameter=0.43912793, is_variational=True))
        circuit.add_gate(Gate("CNOT", 1, 0))
        circuit.add_gate(Gate("RX", 1, parameter=10.995574287564))
        circuit.add_gate(Gate("H", 0))

        # No Noise model.
        nmp_no_noise = NoiseModel()
        noise = 0.00
        nmp_no_noise.add_quantum_error("CNOT", "pauli", [noise, noise, noise])
        sim_no_noise = get_backend(target=default_simulator, n_shots=10**6, noise_model=nmp_no_noise)

        # Small Noise model
        nmp_small_noise = NoiseModel()
        noise = 0.01
        nmp_small_noise.add_quantum_error("CNOT", "pauli", [noise, noise, noise])
        sim_small_noise = get_backend(target=default_simulator, n_shots=10**6, noise_model=nmp_small_noise)

        energy_no_noise = sim_no_noise.get_expectation_value(H, circuit)
        energy_small_noise = sim_small_noise.get_expectation_value(H, circuit)

        self.assertAlmostEqual(energy_no_noise, -14.58922316, delta=1e-2)
        self.assertAlmostEqual(energy_small_noise, -14.58922316, delta=1e-1)


if __name__ == "__main__":
    unittest.main()
