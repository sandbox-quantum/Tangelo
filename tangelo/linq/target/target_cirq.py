# Copyright SandboxAQ 2021-2024.
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

from tangelo.linq import Circuit, Gate, get_unitary_circuit_pieces
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
        n_cmeas = source_circuit.counts.get("CMEASURE", 0)

        if self._noise_model and n_cmeas > 0:
            raise NotImplementedError(f"{self.__class__.__name__} does not currently support measurement-controlled"
                               "gates with a noise model.")

        # Only DensityMatrixSimulator handles noise well, can use Simulator, but it is slower.
        # When not saving mid-circuit measurements, the measurement operation can be converted to a noise channel
        # such that the circuit can be simulated once and the density matrix sampled at the end.
        if self._noise_model or (source_circuit.is_mixed_state and not save_mid_circuit_meas):
            cirq_simulator = self.cirq.DensityMatrixSimulator(dtype=np.complex128)
        else:
            cirq_simulator = self.cirq.Simulator(dtype=np.complex128)

        # If requested, set initial state
        cirq_initial_statevector = np.asarray(initial_statevector, dtype=complex) if initial_statevector is not None else 0

        # If a CMEASURE (measurement-controlled) gate is present, use this branch for all cases.
        # Can be refactored later if/when cirq supports classically controlled gates.
        if n_cmeas > 0:

            self.all_frequencies = dict()
            samples = dict()
            n_shots = self.n_shots if self.n_shots is not None else 1
            n_qubits = source_circuit.width
            indices = list(range(n_qubits))

            for _ in range(n_shots):

                if initial_statevector is not None:
                    sv = cirq_initial_statevector
                else:
                    sv = np.zeros(2**source_circuit.width)
                    sv[0] = 1
                success_probability = 1.
                applied_gates = []
                dmeas = None if not desired_meas_result else list(desired_meas_result)
                measurements = ""

                # Break circuit into pieces that do not include CMEASURE or MEASURE gates
                unitary_circuits, qubits, cmeasure_flags = get_unitary_circuit_pieces(source_circuit)
                # Generate list of circuits that are extended by previous CMEASURE operations
                precirc = [Circuit()]*len(unitary_circuits)

                # CMEASURE operations can return gates that include CMEASURE operations.
                # Example: repeat until success circuits.
                # Therefore, the list of unitary_circuits can grow. Apply all unitary pieces and delete from list
                # until all non-measurement circuits segments have been applied.
                while len(unitary_circuits) > 1:
                    c = precirc[0]+unitary_circuits[0]
                    applied_gates += c._gates

                    if c.size > 0:
                        translated_circuit = translate_c(c, "cirq", output_options={"save_measurements": True})
                        job_sim = cirq_simulator.simulate(translated_circuit, initial_state=sv)
                        sv = job_sim.final_state_vector

                    # Perform measurement.
                    desired_meas = dmeas[0] if desired_meas_result else None
                    measure, sv, cprob = self.perform_measurement(sv, qubits[0], desired_meas)
                    measurements += measure
                    success_probability *= cprob
                    if desired_meas_result:
                        del dmeas[0]

                    # If a CMEASURE has occurred
                    if cmeasure_flags[0] is not None:
                        applied_gates += [Gate("CMEASURE", qubits[0], parameter=measure)]
                        if isinstance(cmeasure_flags[0], str):
                            newcirc = source_circuit.controlled_measurement_op(measure)
                        elif isinstance(cmeasure_flags[0], dict):
                            newcirc = Circuit(cmeasure_flags[0][measure], n_qubits=source_circuit.width)
                        new_unitary_circuits, new_qubits, new_cmeasure_flags = get_unitary_circuit_pieces(newcirc)

                    # No classical control
                    else:
                        applied_gates += [Gate("MEASURE", qubits[0], parameter=measure)]
                        new_unitary_circuits = [Circuit(n_qubits=source_circuit.width)]
                        new_qubits = []
                        new_cmeasure_flags = []

                    # Remove circuits, measurements and corresponding qubits that have been applied.
                    del unitary_circuits[0]
                    del qubits[0]
                    del cmeasure_flags[0]
                    del precirc[0]
                    precirc[0] = new_unitary_circuits[-1] + precirc[0]

                    # If new_unitary_circuits includes MEASURE or CMEASURE Gates, the number of unitary_circuits
                    # grows.
                    if len(new_unitary_circuits) > 1:
                        unitary_circuits = new_unitary_circuits[:-1] + unitary_circuits
                        qubits = new_qubits + qubits
                        cmeasure_flags = new_cmeasure_flags + cmeasure_flags
                        precirc = [Circuit()]*len(qubits) + precirc

                # No more MEASURE or CMEASURE gates are present, run final unitary circuit segment and set attributes
                final_circuit = precirc[0] + unitary_circuits[-1]
                translated_circuit = translate_c(final_circuit, "cirq", output_options={"save_measurements": True})
                job_sim = cirq_simulator.simulate(translated_circuit, initial_state=sv)

                self._current_state = job_sim.final_state_vector
                source_circuit._probabilities[measurements] = success_probability
                source_circuit._applied_gates = applied_gates + final_circuit._gates

                # Update counts if n_shots are required
                if self.n_shots is None:
                    frequencies = self._statevector_to_frequencies(self._current_state)
                    for meas, val in frequencies.items():
                        self.all_frequencies[measurements + meas] = val

                # Obtain full dictionary of frequencies from the final statevector.
                else:
                    isamples = self.cirq.sample_state_vector(self._current_state, indices, repetitions=1)
                    bitstr = measurements + "".join([str(int(q)) for q in isamples[0]])
                    samples[bitstr] = samples.get(bitstr, 0) + 1

                # Call the finalize method of ClassicalControl, used to reset variables, perform computation etc.
                source_circuit.finalize_cmeasure_control()

            # Convert counts to frequencies
            if self.n_shots:
                self.all_frequencies = {k: v / self.n_shots for k, v in samples.items()}
                frequencies = {k[:]: v / self.n_shots for k, v in samples.items()}

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        elif (self._noise_model or source_circuit.is_mixed_state) and not save_mid_circuit_meas:
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
        elif save_mid_circuit_meas and not return_statevector and self.n_shots is not None and n_cmeas == 0:
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

                unitary_circuits, qubits, _ = get_unitary_circuit_pieces(source_circuit)

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
