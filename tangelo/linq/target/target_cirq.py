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

from collections import Counter

import numpy as np

from tangelo.linq import Circuit, get_unitary_circuit_pieces
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator import translate_operator


class CirqSimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        """Instantiate cirq simulator object.

        Args:
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format
                expected from the target backend.
        """
        import cirq
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.cirq = cirq

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
        n_meas = source_circuit.counts.get("MEASURE", 0)
        # Only DensityMatrixSimulator handles noise well, can use Simulator, but it is slower
        if self._noise_model or (source_circuit.is_mixed_state and not save_mid_circuit_meas):
            cirq_simulator = self.cirq.DensityMatrixSimulator(dtype=np.complex128)
        else:
            cirq_simulator = self.cirq.Simulator(dtype=np.complex128)

        # If requested, set initial state
        cirq_initial_statevector = np.asarray(initial_statevector, dtype=complex) if initial_statevector is not None else 0

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        if (self._noise_model or source_circuit.is_mixed_state) and not save_mid_circuit_meas:
            translated_circuit = translate_c(source_circuit, "cirq", output_options={"noise_model": self._noise_model,
                                                                                     "save_measurements": save_mid_circuit_meas})
            # cirq.dephase_measurements changes measurement gates to Krauss operators so simulators
            # can be called once and density matrix sampled repeatedly.
            translated_circuit = self.cirq.dephase_measurements(translated_circuit)
            sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = sim.final_density_matrix
            indices = list(range(source_circuit.width))
            isamples = self.cirq.sample_density_matrix(sim.final_density_matrix, indices, repetitions=self.n_shots)
            samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]
            frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}

        # Run all shots at once and post-process to return measured frequencies on qubits only
        elif save_mid_circuit_meas and not return_statevector and self.n_shots is not None:
            translated_circuit = translate_c(source_circuit, "cirq", output_options={"noise_model": self._noise_model,
                                                                                     "save_measurements": True})
            qubit_list = self.cirq.LineQubit.range(source_circuit.width)
            for i, qubit in enumerate(qubit_list):
                translated_circuit.append(self.cirq.measure(qubit, key=str(i + n_meas)))
            job_sim = cirq_simulator.run(translated_circuit, repetitions=self.n_shots)
            samples = dict()
            for j in range(self.n_shots):
                bitstr = "".join([str(job_sim.measurements[str(i)][j, 0]) for i in range(n_meas + source_circuit.width)])
                samples[bitstr] = samples.get(bitstr, 0) + 1
            self.all_frequencies = {k: v / self.n_shots for k, v in samples.items()}
            frequencies = self.all_frequencies

        # Run shot by shot and keep track of desired_meas_result only if n_shots is set
        # Otherwised, Split circuit into chunks between mid-circuit measurements. Simulate a chunk, collapse the statevector according
        # to the desired measurement and simulate the next chunk using this new statevector as input
        elif desired_meas_result or (save_mid_circuit_meas and return_statevector):

            # desired_meas_result without a noise model
            if self.n_shots is None:
                success_probability = 1
                if initial_statevector is not None:
                    sv = cirq_initial_statevector
                else:
                    sv = np.zeros(2**source_circuit.width)
                    sv[0] = 1

                unitary_circuits, qubits = get_unitary_circuit_pieces(source_circuit)

                for i, circ in enumerate(unitary_circuits[:-1]):
                    if circ.size > 0:
                        translated_circuit = translate_c(circ, "cirq", output_options={"save_measurements": True})
                        job_sim = cirq_simulator.simulate(translated_circuit, initial_state=sv)
                        sv, cprob = self.collapse_statevector_to_desired_measurement(job_sim.final_state_vector, qubits[i], int(desired_meas_result[i]))
                    else:
                        sv, cprob = self.collapse_statevector_to_desired_measurement(sv, qubits[i], int(desired_meas_result[i]))
                    success_probability *= cprob
                source_circuit._probabilities[desired_meas_result] = success_probability

                translated_circuit = translate_c(unitary_circuits[-1], "cirq", output_options={"save_measurements": True})
                job_sim = cirq_simulator.simulate(translated_circuit, initial_state=sv)
                self._current_state = job_sim.final_state_vector
            # Either desired_meas_result with noise_model. Or 1 shot save_mid_circuit_meas
            else:
                translated_circuit = translate_c(source_circuit, "cirq", output_options={"noise_model": self._noise_model,
                                                                                         "save_measurements": True})
                self._current_state = None
                indices = list(range(source_circuit.width))

            # Permit 0.1% probability events
            n_attempts = 0
            max_attempts = 1000 if self.n_shots is None else 1000*self.n_shots

            # Use density matrix simulator until success if noise_model.
            # TODO: implement collapse operations for density matrix simulation.
            # Loop also used for 1 shot if no desired_meas_result and save_mid_circuit_meas.
            while self._current_state is None and n_attempts < max_attempts:
                job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                measure = "".join([str(job_sim.measurements[str(i)][0]) for i in range(n_meas)])
                current_state = job_sim.final_density_matrix if self._noise_model else job_sim.final_state_vector
                if measure == desired_meas_result or desired_meas_result is None:
                    self._current_state = current_state
                n_attempts += 1

            if self._current_state is None:
                raise ValueError(f"desired_meas_result was not measured after {n_attempts} attempts")

            if self.n_shots:
                isamples = (self.cirq.sample_density_matrix(self._current_state, indices, repetitions=self.n_shots) if self._noise_model
                            else self.cirq.sample_state_vector(self._current_state, indices, repetitions=self.n_shots))
                samples = dict()
                for i in range(self.n_shots):
                    sample = "".join([str(int(q)) for q in isamples[i]])
                    bitstr = measure + sample
                    samples[bitstr] = samples.get(bitstr, 0) + 1
                self.all_frequencies = {k: v / self.n_shots for k, v in samples.items()}
            # Noiseless simulation using the statevector simulator otherwise
            else:
                frequencies = self._statevector_to_frequencies(self._current_state)
                self.all_frequencies = dict()
                for meas, val in frequencies.items():
                    self.all_frequencies[desired_meas_result + meas] = val

            if n_attempts == max_attempts:
                raise ValueError(f"desired_meas_result was not measured after {n_attempts} attempts")

            frequencies = self.all_frequencies

        else:
            translated_circuit = translate_c(source_circuit, "cirq", output_options={"noise_model": self._noise_model})
            job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = job_sim.final_state_vector
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):
        """ Compute an expectation value using a representation of the state (density matrix, state vector...)
        using Cirq functionalities.

        Args:
            qubit_operator (QubitOperator): a qubit operator in tangelo format
            n_qubits (int): the number of qubits the operator acts on
            prepared_state (np.array): a numpy array encoding the state (can be a vector or a matrix)

        Returns:
            float64 : the expectation value of the qubit operator w.r.t the input state
        """

        # Construct equivalent Pauli operator in Cirq format
        qubit_labels = self.cirq.LineQubit.range(n_qubits)
        qubit_map = {q: i for i, q in enumerate(qubit_labels)}
        paulisum = translate_operator(qubit_operator, source="tangelo", target="cirq")

        # Compute expectation value using Cirq's features
        if self._noise_model:
            exp_value = paulisum.expectation_from_density_matrix(prepared_state, qubit_map)
        else:
            exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
        return np.real(exp_value)

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}
