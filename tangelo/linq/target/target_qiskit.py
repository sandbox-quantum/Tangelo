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

import math
from collections import Counter

import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c


class QiskitSimulator(Backend):
    """Interface to the qiskit simulator."""

    def __init__(self, n_shots=None, noise_model=None):
        import qiskit
        from qiskit_aer import AerSimulator
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.qiskit = qiskit
        self.AerSimulator = AerSimulator

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None,
                         desired_meas_result=None, save_mid_circuit_meas=False):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit (Circuit): a circuit in the abstract format to be translated
                for the target backend.
            return_statevector (bool): option to return the statevector as well,
                if available.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The binary string of the desired measurement.
                Must have the same length as the number of MEASURE gates in source_circuit
            save_mid_circuit_meas (bool): Save mid-circuit measurement results to
                self.mid_circuit_meas_freqs. All measurements will be saved to
                self.all_frequencies.
                self.all_frequencies will have keys of length m + n_qubits
                The first m values in each key will be the result of the m MEASURE gates ordered
                by their appearance in the list of gates in the source_circuit.
                The last n_qubits values in each key will be the measurements performed on
                each of qubits at the end of the circuit.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """

        translated_circuit = translate_c(source_circuit, "qiskit", output_options={"save_measurements": save_mid_circuit_meas})

        # If requested, set initial state
        if initial_statevector is not None:
            if self._noise_model:
                raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")
            else:
                n_qubits = int(math.log2(len(initial_statevector)))
                n_meas = source_circuit._gate_counts.get("MEASURE", 0)
                n_registers = n_meas + source_circuit.width if save_mid_circuit_meas else source_circuit.width
                initial_state_circuit = self.qiskit.QuantumCircuit(n_qubits, n_registers)
                initial_state_circuit.initialize(initial_statevector, list(range(n_qubits)))
                translated_circuit = initial_state_circuit.compose(translated_circuit)

        # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state and (desired_meas_result is None or not return_statevector):
            from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model

            n_meas = source_circuit._gate_counts.get("MEASURE", 0)
            meas_start = n_meas if save_mid_circuit_meas else 0
            meas_range = range(meas_start, meas_start + source_circuit.width)
            translated_circuit.measure(range(source_circuit.width), meas_range)
            return_statevector = False
            backend = self.AerSimulator()

            qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
            opt_level = 0 if self._noise_model else None

            job_sim = self.qiskit.execute(translated_circuit, backend, noise_model=qiskit_noise_model,
                                          shots=self.n_shots, basis_gates=None, optimization_level=opt_level)
            sim_results = job_sim.result()
            frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

            self.all_frequencies = frequencies.copy()
            if source_circuit.is_mixed_state and save_mid_circuit_meas:
                from tangelo.toolboxes.post_processing.post_selection import split_frequency_dict

                self.mid_circuit_meas_freqs, frequencies = split_frequency_dict(self.all_frequencies,
                                                                                list(range(n_meas)),
                                                                                desired_measurement=desired_meas_result)
            self._current_state = None

        # desired_meas_result is not None and return_statevector is requested so loop shot by shot (much slower)
        elif desired_meas_result is not None:
            from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model
            n_meas = source_circuit._gate_counts.get("MEASURE", 0)
            backend = self.AerSimulator(method='statevector')
            qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
            translated_circuit = self.qiskit.transpile(translated_circuit, backend)
            translated_circuit.save_statevector()
            self.mid_circuit_meas_freqs = dict()
            self.all_frequencies = dict()
            samples = list()
            successful_measures = 0
            self._current_state = None

            for _ in range(self.n_shots):
                sim_results = backend.run(translated_circuit, noise_model=qiskit_noise_model, shots=1).result()
                current_state = sim_results.get_statevector(translated_circuit)
                measurement = next(
                    iter(self.qiskit.result.marginal_counts(sim_results, indices=list(range(n_meas))).get_counts()))[::-1]
                (sample, _) = self.qiskit.quantum_info.states.Statevector(current_state).measure()
                key = measurement + sample[::-1]
                self.all_frequencies[key] = self.all_frequencies.get(key, 0) + 1
                self.mid_circuit_meas_freqs[measurement] = self.mid_circuit_meas_freqs.get(measurement, 0) + 1
                if measurement == desired_meas_result:
                    self._current_state = current_state
                    successful_measures += 1
                    samples += [sample[::-1]]
            self.all_frequencies = {k: v / self.n_shots for k, v in self.all_frequencies.items()}
            self.mid_circuit_meas_freqs = {k: v / self.n_shots for k, v in self.mid_circuit_meas_freqs.items()}
            frequencies = {k: v / successful_measures for k, v in Counter(samples).items()}
            self.success_probability = successful_measures / self.n_shots

        # Noiseless simulation using the statevector simulator otherwise
        else:
            backend = self.AerSimulator(method='statevector')
            translated_circuit = self.qiskit.transpile(translated_circuit, backend)
            translated_circuit.save_statevector()
            sim_results = backend.run(translated_circuit).result()
            self._current_state = np.asarray(sim_results.get_statevector(translated_circuit))
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if (return_statevector and self._current_state is not None) else (frequencies, None)

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
