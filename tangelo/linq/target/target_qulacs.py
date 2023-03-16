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


import os
from collections import Counter

import numpy as np

from tangelo.linq import Circuit, get_unitary_circuit_pieces
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator import translate_operator


class QulacsSimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        import qulacs
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.qulacs = qulacs

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
                self.all_frequencies, with keys of length (n_meas + n_qubits).
                The leading n_meas values will hold the results of the MEASURE gates,
                ordered by their appearance in the source_circuit.
                The last n_qubits values will hold the measurements performed on
                each of qubits at the end of the circuit.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        if desired_meas_result is None:
            translated_circuit = translate_c(source_circuit, "qulacs", output_options={"noise_model": self._noise_model,
                                                                                       "save_measurements": save_mid_circuit_meas})

        n_meas = source_circuit.counts.get("MEASURE", 0)

        # Initialize state on GPU if available and desired. Default to CPU otherwise.
        if ('QuantumStateGpu' in dir(self.qulacs)) and (int(os.getenv("QULACS_USE_GPU", 0)) != 0):
            state = self.qulacs.QuantumStateGpu(source_circuit.width)
        else:
            state = self.qulacs.QuantumState(source_circuit.width)

        python_statevector = None
        if initial_statevector is not None:
            state.load(initial_statevector)

        # Deterministic circuit, run once and sample from statevector
        if not self._noise_model and not source_circuit.is_mixed_state:
            translated_circuit.update_quantum_state(state)
            self._current_state = state

            if self.n_shots is not None:
                python_statevector = np.array(state.get_vector()) if return_statevector else None
                samples = Counter(state.sampling(self.n_shots))  # this sampling still returns a list
            else:
                python_statevector = np.array(state.get_vector())
                frequencies = self._statevector_to_frequencies(python_statevector)
                return (frequencies, python_statevector) if return_statevector else (frequencies, None)

        # If a desired_meas_result,
        # If no noise model, Split circuit into chunks between mid-circuit measurements. Simulate a chunk, collapse the statevector according
        # to the desired measurement and simulate the next chunk using this new statevector as input
        # If noise_model, repeat until n_shots desired_meas_results.
        elif desired_meas_result is not None:
            if not self._noise_model:
                success_probability = 1
                unitary_circuits, qubits = get_unitary_circuit_pieces(source_circuit)

                for i, circ in enumerate(unitary_circuits[:-1]):
                    if circ.size > 0:
                        translated_circuit = translate_c(circ, "qulacs", output_options={"save_measurements": True})
                        translated_circuit.update_quantum_state(state)
                    sv, cprob = self.collapse_statevector_to_desired_measurement(state.get_vector(), qubits[i], int(desired_meas_result[i]))
                    success_probability *= cprob
                    state.load(sv)

                translated_circuit = translate_c(unitary_circuits[-1], "qulacs", output_options={"save_measurements": True})
                translated_circuit.update_quantum_state(state)
                python_statevector = state.get_vector()
                self._current_state = python_statevector
                source_circuit._probabilities[desired_meas_result] = success_probability

                if self.n_shots is not None:
                    samples = Counter(state.sampling(self.n_shots))
                else:
                    frequencies = self._statevector_to_frequencies(python_statevector)
                    self.all_frequencies = dict()
                    for meas, val in frequencies.items():
                        self.all_frequencies[desired_meas_result + meas] = val
            else:
                translated_circuit = translate_c(source_circuit, "qulacs", output_options={"noise_model": self._noise_model,
                                                                                           "save_measurements": save_mid_circuit_meas})
                self._current_state = None

                n_success = 0

                # Permit 0.1% probability events
                n_attempts = 0
                max_attempts = 1000*self.n_shots
                samples = dict()

                while self._current_state is None and n_attempts < max_attempts and n_success != self.n_shots:
                    translated_circuit.update_quantum_state(state)
                    measure = "".join([str(state.get_classical_value(i)) for i in range(n_meas)])

                    if measure == desired_meas_result:
                        python_statevector = state.get_vector()
                        if self._noise_model:
                            n_success += 1
                            sample = state.sampling(1)[0]
                            samples[sample] = samples.get(sample, 0) + 1

                    if initial_statevector is not None:
                        state.load(initial_statevector)
                    else:
                        state.set_zero_state()
                    n_attempts += 1

                if n_attempts == max_attempts:
                    raise ValueError(f"desired_meas_result was not measured after {n_attempts} attempts")

            if self.n_shots is not None:
                self.all_frequencies = {desired_meas_result + self._int_to_binstr(k, source_circuit.width): v / self.n_shots
                                        for k, v in samples.items()}

            return (self.all_frequencies, python_statevector) if return_statevector else (self.all_frequencies, None)

        # No desired_meas_result, repeat simulation n_shot times and collect measurement data.
        # Same process for if noise model present or not.
        elif save_mid_circuit_meas:
            samples = dict()
            for _ in range(self.n_shots):
                translated_circuit.update_quantum_state(state)
                measure = "".join([str(state.get_classical_value(i)) for i in range(n_meas)])
                sample = self._int_to_binstr(state.sampling(1)[0], source_circuit.width)
                bitstr = measure + sample
                samples[bitstr] = samples.get(bitstr, 0) + 1

                # Will only be true if self.n_shots=1
                if return_statevector:
                    python_statevector = state.get_vector()
                    self._current_state = python_statevector
                if initial_statevector is not None:
                    state.load(initial_statevector)
                else:
                    state.set_zero_state()

            self.all_frequencies = {k: v / self.n_shots for k, v in samples.items()}
            return (self.all_frequencies, python_statevector) if return_statevector else (self.all_frequencies, None)

        # Not saving mid-circuit measurements or a noise model, repeat simulation n_shots times.
        elif source_circuit.is_mixed_state or self._noise_model:
            samples = dict()
            for _ in range(self.n_shots):
                translated_circuit.update_quantum_state(state)
                bitstr = state.sampling(1)[0]
                samples[bitstr] = samples.get(bitstr, 0) + 1
                if initial_statevector is not None:
                    state.load(initial_statevector)
                else:
                    state.set_zero_state()

        frequencies = {self._int_to_binstr(k, source_circuit.width): v / self.n_shots
                       for k, v in samples.items()}
        return (frequencies, python_statevector) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state=None):
        """ Compute an expectation value using a representation of the state
        using qulacs functionalities.

        Args:
            qubit_operator (QubitOperator): a qubit operator in tangelo format
            n_qubits (int): Number of qubits.
            prepared_state (np.array): a numpy array encoding the state (can be
                a vector or a matrix). It is internally transformed into a
                qulacs.QuantumState object. Default is None, in this case it is
                set to the current state in the simulator object.

        Returns:
            float64 : the expectation value of the qubit operator w.r.t the input state
        """
        if prepared_state is None:
            prepared_state = self._current_state
        else:
            qulacs_state = self.qulacs.QuantumState(n_qubits)
            qulacs_state.load(prepared_state)
            prepared_state = qulacs_state

        operator = translate_operator(qubit_operator, source="tangelo", target="qulacs")
        return operator.get_expectation_value(prepared_state).real

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
