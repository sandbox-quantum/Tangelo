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

"""Simulator class, wrapping around the various simulators and abstracting their
differences from the user. Able to run noiseless and noisy simulations,
leveraging the capabilities of different backends, quantum or classical.

If the user provides a noise model, then a noisy simulation is run with n_shots
shots. If the user only provides n_shots, a noiseless simulation is run, drawing
the desired amount of shots. If the target backend has access to the statevector
representing the quantum state, we leverage any kind of emulation available to
reduce runtime (directly generating shot values from final statevector etc) If
the quantum circuit contains a MEASURE instruction, it is assumed to simulate a
mixed-state and the simulation will be carried by simulating individual shots
(e.g a number of shots is required).

Some backends may only support a subset of the above. This information is
contained in a separate data-structure.
"""

import os
import math
from collections import Counter

import numpy as np
from scipy import stats
from bitarray import bitarray
from openfermion.ops import QubitOperator

from tangelo.helpers.utils import default_simulator
from tangelo.linq import Gate, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
import tangelo.linq.translator as translator


# Data-structure showing what functionalities are supported by the backend, in this package
backend_info = dict()
backend_info["qiskit"] = {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
backend_info["qulacs"] = {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
backend_info["cirq"] = {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}
backend_info["qdk"] = {"statevector_available": False, "statevector_order": None, "noisy_simulation": False}


class Simulator:

    def __init__(self, target=default_simulator, n_shots=None, noise_model=None):
        """Instantiate Simulator object.

        Args:
            target (str): One of the available target backends (quantum or
                classical). The default simulator is qulacs if found in user's
                environment, otherwise cirq.
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format
                expected from the target backend.
        """
        self._source = "abstract"
        self._target = target if target else default_simulator
        self._current_state = None
        self._noise_model = noise_model

        # Can be modified later by user as long as long as it retains the same type (ex: cannot change to/from None)
        self.n_shots = n_shots
        self.freq_threshold = 1e-10

        # Set additional attributes related to the target backend chosen by the user
        for k, v in backend_info[self._target].items():
            setattr(self, k, v)

        # Raise error if user attempts to pass a noise model to a backend not supporting noisy simulation
        if self._noise_model and not self.noisy_simulation:
            raise ValueError("Target backend does not support noise models.")

        # Raise error if the number of shots has not been passed for a noisy simulation or if statevector unavailable
        if not self.n_shots and (not self.statevector_available or self._noise_model):
            raise ValueError("A number of shots needs to be specified.")

    def simulate(self, source_circuit, return_statevector=False, initial_statevector=None,
                 desired_meas_result=None, save_mid_circuit_meas=False):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit: a circuit in the abstract format to be translated
                for the target backend.
            return_statevector(bool): option to return the statevector as well,
                if available.
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The binary string of the desired measurement.
                Must have the same length as the number of MEASURE gates in circuit
            save_mid_circuit_meas (bool): Save mid-circuit measurement results to
                self.mid_circuit_meas_freqs. All measurements will be save to
                self.all_frequencies

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """

        if source_circuit.is_mixed_state and not self.n_shots:
            raise ValueError("Circuit contains MEASURE instruction, and is assumed to prepare a mixed state."
                             "Please set the Simulator.n_shots attribute to an appropriate value.")

        n_meas = source_circuit._gate_counts.get("MEASURE", 0)
        if desired_meas_result is not None:
            if len(desired_meas_result) != n_meas or not isinstance(desired_meas_result, str):
                raise ValueError("desired_meas result is not a string with the same length as the number of measurements"
                                 "in the circuit.")
            save_mid_circuit_meas = True

        if source_circuit.width == 0:
            raise ValueError("Cannot simulate an empty circuit (e.g identity unitary) with unknown number of qubits.")

        # If the unitary is the identity (no gates) and no noise model, no need for simulation:
        # return all-zero state or sample from statevector
        if source_circuit.size == 0 and not self._noise_model:
            if initial_statevector is not None:
                statevector = initial_statevector
                frequencies = self._statevector_to_frequencies(initial_statevector)
            else:
                frequencies = {'0'*source_circuit.width: 1.0}
                statevector = np.zeros(2**source_circuit.width)
                statevector[0] = 1.0
            return (frequencies, statevector) if return_statevector else (frequencies, None)

        if self._target == "qulacs":
            import qulacs

            translated_circuit = translator.translate_qulacs(source_circuit, self._noise_model)

            # Initialize state on GPU if available and desired. Default to CPU otherwise.
            if ('QuantumStateGpu' in dir(qulacs)) and (int(os.getenv("QULACS_USE_GPU", 0)) != 0):
                state = qulacs.QuantumStateGpu(source_circuit.width)
            else:
                state = qulacs.QuantumState(source_circuit.width)
            if initial_statevector is not None:
                state.load(initial_statevector)

            if (source_circuit.is_mixed_state or self._noise_model) and (desired_meas_result is None and not save_mid_circuit_meas):
                samples = list()
                for i in range(self.n_shots):
                    translated_circuit.update_quantum_state(state)
                    samples.append(state.sampling(1)[0])
                    if initial_statevector is not None:
                        state.load(initial_statevector)
                    else:
                        state.set_zero_state()
                python_statevector = None
            elif desired_meas_result is not None or save_mid_circuit_meas:
                samples = list()
                full_samples = list()
                successful_measures = 0 if desired_meas_result is not None else self.n_shots
                measurements = list()
                python_statevector = None
                for _ in range(self.n_shots):
                    translated_circuit.update_quantum_state(state)
                    measurement = "".join([str(state.get_classical_value(i)) for i in range(n_meas)])
                    measurements.append(measurement)
                    sample = self.__int_to_binstr(state.sampling(1)[0], source_circuit.width)
                    full_samples += [measurement+sample]
                    if desired_meas_result is not None:
                        if measurement == desired_meas_result:
                            samples.append(sample)
                            successful_measures += 1
                            if return_statevector and python_statevector is None:
                                python_statevector = state.get_vector()
                    else:
                        samples.append(sample)
                    if initial_statevector is not None:
                        state.load(initial_statevector)
                    else:
                        state.set_zero_state()
                self.all_frequencies = {k: v / self.n_shots for k, v in Counter(full_samples).items()}
                self.mid_circuit_meas_freqs = {k: v / self.n_shots for k, v in Counter(measurements).items()}
                frequencies = {k: v / successful_measures for k, v in Counter(samples).items()}
                self.success_probability = successful_measures / self.n_shots
                return (frequencies, python_statevector)
            elif self.n_shots is not None:
                translated_circuit.update_quantum_state(state)
                python_statevector = np.array(state.get_vector()) if return_statevector else None
                samples = state.sampling(self.n_shots)
            else:
                translated_circuit.update_quantum_state(state)
                self._current_state = state
                python_statevector = state.get_vector()
                frequencies = self._statevector_to_frequencies(python_statevector)
                return (frequencies, np.array(python_statevector)) if return_statevector else (frequencies, None)

            frequencies = {self.__int_to_binstr(k, source_circuit.width): v / self.n_shots
                           for k, v in Counter(samples).items()}
            return (frequencies, python_statevector)

        elif self._target == "qiskit":
            import qiskit

            translated_circuit = translator.translate_qiskit(source_circuit, save_measurements=save_mid_circuit_meas)

            # If requested, set initial state
            if initial_statevector is not None:
                if self._noise_model:
                    raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")
                else:
                    n_qubits = int(math.log2(len(initial_statevector)))
                    n_registers = source_circuit._gate_counts.get("MEASURE", 0) + source_circuit.width
                    initial_state_circuit = qiskit.QuantumCircuit(n_qubits, n_registers)
                    initial_state_circuit.initialize(initial_statevector, list(range(n_qubits)))
                    translated_circuit = initial_state_circuit.compose(translated_circuit)

            # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
            if self._noise_model or source_circuit.is_mixed_state and (desired_meas_result is None or not return_statevector):
                from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model
                meas_start = n_meas if save_mid_circuit_meas else 0
                meas_range = range(meas_start, meas_start+source_circuit.width)
                translated_circuit.measure(range(source_circuit.width), meas_range)
                return_statevector = False
                backend = qiskit.Aer.get_backend("aer_simulator")

                qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
                opt_level = 0 if self._noise_model else None

                job_sim = qiskit.execute(translated_circuit, backend, noise_model=qiskit_noise_model,
                                         shots=self.n_shots, basis_gates=None, optimization_level=opt_level)
                sim_results = job_sim.result()
                frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}
                self.all_frequencies = frequencies.copy()
                if source_circuit.is_mixed_state and save_mid_circuit_meas:
                    self.mid_circuit_meas_freqs, frequencies = self.marginal_frequencies(self.all_frequencies,
                                                                                         list(range(n_meas)),
                                                                                         desired_measurement=desired_meas_result)
                self._current_state = None
            # desired_meas_result is not None and return_statevector is requested so loop shot by shot (much slower)
            elif desired_meas_result is not None:
                from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model
                backend = qiskit.Aer.get_backend("aer_simulator", method='statevector')
                qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
                opt_level = 0 if self._noise_model else None
                translated_circuit = qiskit.transpile(translated_circuit, backend)
                translated_circuit.save_statevector()
                self.mid_circuit_meas_freqs = dict()
                self.all_frequencies = dict()
                samples = list()
                successful_measures = 0
                self._current_state = None

                for _ in range(self.n_shots):
                    sim_results = backend.run(translated_circuit, noise_model=qiskit_noise_model, shots=1).result()
                    current_state = sim_results.get_statevector(translated_circuit)
                    measurement = next(iter(qiskit.result.marginal_counts(sim_results, indices=list(range(n_meas))).get_counts()))[::-1]
                    (sample, _) = qiskit.quantum_info.states.Statevector(current_state).measure()
                    key = measurement+sample[::-1]
                    self.all_frequencies[key] = self.all_frequencies.get(key, 0) + 1
                    self.mid_circuit_meas_freqs[measurement] = self.mid_circuit_meas_freqs.get(measurement, 0) + 1
                    if measurement == desired_meas_result:
                        self._current_state = current_state
                        successful_measures += 1
                        samples += [sample[::-1]]
                self.all_frequencies = {k: v / self.n_shots for k, v in self.all_frequencies.items()}
                self.mid_circuit_meas_freqs = {k: v / self.n_shots for k, v in self.mid_circuit_meas_freqs.items()}
                frequencies = {k: v/successful_measures for k, v in Counter(samples).items()}
                self.success_probability = successful_measures / self.n_shots
            # Noiseless simulation using the statevector simulator otherwise
            else:
                backend = qiskit.Aer.get_backend("aer_simulator", method='statevector')
                translated_circuit = qiskit.transpile(translated_circuit, backend)
                translated_circuit.save_statevector()
                sim_results = backend.run(translated_circuit).result()
                self._current_state = sim_results.get_statevector(translated_circuit)
                frequencies = self._statevector_to_frequencies(self._current_state)

            return (frequencies, np.array(self._current_state)) if (return_statevector and self._current_state is not None) else (frequencies, None)

        elif self._target == "qdk":

            translated_circuit = translator.translate_qsharp(source_circuit, save_measurements=save_mid_circuit_meas)
            with open('tmp_circuit.qs', 'w+') as f_out:
                f_out.write(translated_circuit)

            key_length = n_meas + source_circuit.width if save_mid_circuit_meas else source_circuit.width
            # Compile, import and call Q# operation to compute frequencies. Only import qsharp module if qdk is running
            # TODO: A try block to catch an exception at compile time, for Q#? Probably as an ImportError.
            import qsharp
            qsharp.reload()
            from MyNamespace import EstimateFrequencies
            frequencies_list = EstimateFrequencies.simulate(nQubits=key_length, nShots=self.n_shots)
            print("Q# frequency estimation with {0} shots: \n {1}".format(self.n_shots, frequencies_list))

            # Convert Q# output to frequency dictionary, apply threshold
            frequencies = {bin(i).split('b')[-1]: frequencies_list[i] for i, freq in enumerate(frequencies_list)}
            frequencies = {("0"*(key_length-len(k))+k)[::-1]: v for k, v in frequencies.items()
                           if v > self.freq_threshold}
            self.all_frequencies = frequencies.copy()
            # Post process if needed
            if save_mid_circuit_meas:
                self.mid_circuit_meas_freqs, frequencies = self.marginal_frequencies(self.all_frequencies,
                                                                                     list(range(n_meas)),
                                                                                     desired_measurement=desired_meas_result)
            return (frequencies, None)

        elif self._target == "cirq":
            import cirq
            from cirq.transformers.measurement_transformers import dephase_measurements

            if (source_circuit.is_mixed_state or self._noise_model) and not save_mid_circuit_meas:
                # Only DensityMatrixSimulator handles noise well, can use Simulator but it is slower
                # ignore_measurement_results changes measurement gates to Krauss operators so simulators
                # can be called once and density matrix sampled repeatedly.
                cirq_simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
            elif save_mid_circuit_meas:
                cirq_simulator = cirq.DensityMatrixSimulator(dtype=np.complex128) if self._noise_model else cirq.Simulator(dtype=np.complex128)
            else:
                cirq_simulator = cirq.Simulator(dtype=np.complex128)

            # If requested, set initial state
            cirq_initial_statevector = initial_statevector if initial_statevector is not None else 0

            # Calculate final density matrix and sample from that for noisy simulation or simulating non-saved mixed states
            if (self._noise_model or source_circuit.is_mixed_state) and not save_mid_circuit_meas:
                translated_circuit = translator.translate_cirq(source_circuit, self._noise_model)
                translated_circuit = dephase_measurements(translated_circuit)
                sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                self._current_state = sim.final_density_matrix
                indices = list(range(source_circuit.width))
                isamples = cirq.sample_density_matrix(sim.final_density_matrix, indices, repetitions=self.n_shots)
                samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]

                frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
            # Run all shots at once and post-process to return measured frequencies on qubits only
            elif save_mid_circuit_meas and not return_statevector:
                translated_circuit = translator.translate_cirq(source_circuit, self._noise_model, save_measurements=True)
                qubit_list = cirq.LineQubit.range(source_circuit.width)
                for i, qubit in enumerate(qubit_list):
                    translated_circuit.append(cirq.measure(qubit, key=str(i+n_meas)))
                job_sim = cirq_simulator.run(translated_circuit, repetitions=self.n_shots)
                samples = list()
                for j in range(self.n_shots):
                    samples += ["".join([str(job_sim.measurements[str(i)][j, 0]) for i in range(n_meas+source_circuit.width)])]
                self.all_frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}

                self.mid_circuit_meas_freqs, frequencies = self.marginal_frequencies(self.all_frequencies,
                                                                                     list(range(n_meas)),
                                                                                     desired_measurement=desired_meas_result)
            # Run shot by shot and keep track of desired_meas_result only (generally slower)
            elif save_mid_circuit_meas and return_statevector:
                translated_circuit = translator.translate_cirq(source_circuit, self._noise_model, save_measurements=True)
                successful_measures = 0 if desired_meas_result is not None else self.n_shots
                samples = list()
                measurements = list()
                all_measurements = list()
                self._current_state = None
                indices = list(range(source_circuit.width))
                for _ in range(self.n_shots):
                    job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                    measure = "".join([str(job_sim.measurements[str(i)][0]) for i in range(n_meas)])
                    measurements.append(measure)
                    current_state = job_sim.final_density_matrix if self._noise_model else job_sim.final_state_vector
                    isamples = (cirq.sample_density_matrix(current_state, indices, repetitions=1) if self._noise_model
                                else cirq.sample_state_vector(current_state, indices, repetitions=1))
                    sample = "".join([str(int(q))for q in isamples[0]])
                    all_measurements += [measure+sample]
                    if measure == desired_meas_result:
                        self._current_state = current_state
                        samples += [sample]
                        successful_measures += 1
                    elif desired_meas_result is None:
                        samples += [sample]
                self.all_frequencies = {k: v / self.n_shots for k, v in Counter(all_measurements).items()}
                frequencies = {k: v / successful_measures for k, v in Counter(samples).items()}
                self.mid_circuit_meas_freqs = {k: v / self.n_shots for k, v in Counter(measurements).items()}
                self.success_probability = successful_measures / self.n_shots
            # Noiseless simulation using the statevector simulator otherwise
            else:
                translated_circuit = translator.translate_cirq(source_circuit, self._noise_model)
                job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                self._current_state = job_sim.final_state_vector
                frequencies = self._statevector_to_frequencies(self._current_state)

            return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    @staticmethod
    def marginal_frequencies(frequencies, indices, desired_measurement=None):
        """Return the marginal frequencies on given indices. If desired_measurement
        is given, frequencies on other indices are conditional on indices measurement being the desired measurement

        Args:
            frequencies (dict): The frequency dictionary to perform the marginal computation
            indices (list): The list of indices in the frequency dictionary to marginalize over
            desired_measurement (str): The bit string that is to be selected

        Returns:
            dict, dict: The marginal frequencies for indices, The marginal frequencies for other indices"""

        new_dict = dict()
        other_dict = dict()
        key_length = len(next(iter(frequencies)))
        other_indices = [i for i in range(key_length) if i not in indices]
        for k, v in frequencies.items():
            new_key = "".join(k[i] for i in indices)
            other_key = "".join(k[i] for i in other_indices)
            new_dict[new_key] = new_dict.get(new_key, 0) + v
            if new_key == desired_measurement:
                other_dict[other_key] = other_dict.get(new_key, 0) + v
            elif desired_measurement is None:
                other_dict[other_key] = other_dict.get(new_key, 0) + v
        if desired_measurement is not None:
            other_dict = {k: v/new_dict[desired_measurement] for k, v in other_dict.items()}

        return new_dict, other_dict

    def get_expectation_value(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a quantum circuit preparing a
        state |\psi>. Return the expectation value <\psi | H | \psi>.

        In the case of a noiseless simulation, if the target backend exposes the
        statevector then it is used directly to compute expectation values, or
        draw samples if required. In the case of a noisy simulator, or if the
        statevector is not available on the target backend, individual shots
        must be run and the workflow is akin to what we would expect from an
        actual QPU.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (array): The initial statevector for the simulation
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        # Check if simulator supports statevector
        if initial_statevector is not None and not self.statevector_available:
            raise ValueError(f'Statevector not supported in {self._target}')

        # Check that qubit operator does not operate on qubits beyond circuit size.
        # Keep track if coefficients are real or not
        are_coefficients_real = True
        for term, coef in qubit_operator.terms.items():
            if state_prep_circuit.width < len(term):
                raise ValueError(f'Term {term} requires more qubits than the circuit contains ({state_prep_circuit.width})')
            if type(coef) in {complex, np.complex64, np.complex128}:
                are_coefficients_real = False

        # If the underlying operator is hermitian, expectation value is real and can be computed right away
        if are_coefficients_real:
            if self._noise_model or not self.statevector_available \
                    or state_prep_circuit.is_mixed_state or state_prep_circuit.size == 0:
                return self._get_expectation_value_from_frequencies(qubit_operator, state_prep_circuit, initial_statevector=initial_statevector,
                                                                    desired_meas_result=desired_meas_result)
            elif self.statevector_available:
                return self._get_expectation_value_from_statevector(qubit_operator, state_prep_circuit, initial_statevector=initial_statevector)

        # Else, separate the operator into 2 hermitian operators, use linearity and call this function twice
        else:
            qb_op_real, qb_op_imag = QubitOperator(), QubitOperator()
            for term, coef in qubit_operator.terms.items():
                qb_op_real.terms[term], qb_op_imag.terms[term] = coef.real, coef.imag
            qb_op_real.compress()
            qb_op_imag.compress()
            exp_real = self.get_expectation_value(qb_op_real, state_prep_circuit, initial_statevector=initial_statevector)
            exp_imag = self.get_expectation_value(qb_op_imag, state_prep_circuit, initial_statevector=initial_statevector)
            return exp_real if (exp_imag == 0.) else exp_real + 1.0j * exp_imag

    def _get_expectation_value_from_statevector(self, qubit_operator, state_prep_circuit, initial_statevector=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi>, computed
        without drawing samples (statevector only). Users should not be calling
        this function directly, please call "get_expectation_value" instead.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation
                (only pure states).
            initial_statevector (array): The initial state of the system

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width

        expectation_value = 0.
        prepared_frequencies, prepared_state = self.simulate(state_prep_circuit, return_statevector=True, initial_statevector=initial_statevector)

        # Use fast built-in qulacs expectation value function if possible
        if self._target == "qulacs" and not self.n_shots:
            import qulacs

            # Note: This section previously used qulacs.quantum_operator.create_quantum_operator_from_openfermion_text but was changed
            # due to a memory leak. We can re-evaluate the implementation if/when Issue #303 (https://github.com/qulacs/qulacs/issues/303)
            # is fixed.
            operator = qulacs.Observable(n_qubits)
            for term, coef in qubit_operator.terms.items():
                pauli_string = "".join(f" {op} {qu}" for qu, op in term)
                operator.add_operator(coef, pauli_string)
            return operator.get_expectation_value(self._current_state).real

        # Use cirq built-in expectation_from_state_vector/epectation_from_density_matrix
        # noise model would require
        if self._target == "cirq" and not self.n_shots:
            import cirq

            GATE_CIRQ = translator.get_cirq_gates()
            qubit_labels = cirq.LineQubit.range(n_qubits)
            qubit_map = {q: i for i, q in enumerate(qubit_labels)}
            paulisum = 0.*cirq.PauliString(cirq.I(qubit_labels[0]))
            for term, coef in qubit_operator.terms.items():
                pauli_list = [GATE_CIRQ[pauli](qubit_labels[index]) for index, pauli in term]
                paulisum += cirq.PauliString(pauli_list, coefficient=coef)
            if self._noise_model:
                exp_value = paulisum.expectation_from_density_matrix(prepared_state, qubit_map)
            else:
                exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
            return np.real(exp_value)

        # Otherwise, use generic statevector expectation value
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:  # Cannot have a qubit index beyond circuit size
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            if not self.n_shots:
                # Directly simulate and compute expectation value using statevector
                pauli_circuit = Circuit([Gate(pauli, index) for index, pauli in term], n_qubits=n_qubits)
                _, pauli_state = self.simulate(pauli_circuit, return_statevector=True, initial_statevector=prepared_state)

                delta = np.dot(pauli_state.real, prepared_state.real) + np.dot(pauli_state.imag, prepared_state.imag)
                expectation_value += coef * delta

            else:
                # Run simulation with statevector but compute expectation value with samples directly drawn from it
                basis_circuit = Circuit(measurement_basis_gates(term), n_qubits=state_prep_circuit.width)
                if basis_circuit.size > 0:
                    frequencies, _ = self.simulate(basis_circuit, initial_statevector=prepared_state)
                else:
                    frequencies = prepared_frequencies
                expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
                expectation_value += coef * expectation_term

        return expectation_value

    def _get_expectation_value_from_frequencies(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi> computed
        using the frequencies of observable states.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (array): The initial state of the system
            desired_meas_result (str): The expectation value is taken for over the frequencies
                derived when the mid-circuit measurements match this string.

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width
        if not self.statevector_available or state_prep_circuit.is_mixed_state or self._noise_model:
            initial_circuit = state_prep_circuit
            if initial_statevector is not None and not self.statevector_available:
                raise ValueError(f'Backend {self._target} does not support statevectors')
            else:
                updated_statevector = initial_statevector
        else:
            initial_circuit = Circuit(n_qubits=n_qubits)
            _, updated_statevector = self.simulate(state_prep_circuit, return_statevector=True, initial_statevector=initial_statevector)

        expectation_value = 0.
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            basis_circuit = Circuit(measurement_basis_gates(term))
            full_circuit = initial_circuit + basis_circuit if (basis_circuit.size > 0) else initial_circuit
            frequencies, _ = self.simulate(full_circuit, initial_statevector=updated_statevector, desired_meas_result=desired_meas_result)
            expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
            expectation_value += coef * expectation_term

        return expectation_value

    @staticmethod
    def get_expectation_value_from_frequencies_oneterm(term, frequencies):
        """Return the expectation value of a single-term qubit-operator, given
        the result of a state-preparation.

        Args:
            term(openfermion-style QubitOperator object): a qubit operator, with
                only a single term.
            frequencies(dict): histogram of frequencies of measurements (assumed
                to be in lsq-first format).

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """

        if not frequencies.keys():
            return ValueError("Must pass a non-empty dictionary of frequencies.")
        n_qubits = len(list(frequencies.keys())[0])

        # Get term mask
        mask = ["0"] * n_qubits
        for index, op in term:
            mask[index] = "1"
        mask = "".join(mask)

        # Compute expectation value of the term
        expectation_term = 0.
        for basis_state, freq in frequencies.items():
            # Compute sample value using state_binstr and term mask, update term expectation value
            sample = (-1) ** ((bitarray(mask) & bitarray(basis_state)).to01().count("1") % 2)
            expectation_term += sample * freq

        return expectation_term

    def _statevector_to_frequencies(self, statevector):
        """For a given statevector representing the quantum state of a qubit
        register, returns a sparse histogram of the probabilities in the
        least-significant-qubit (lsq) -first order. e.g the string '100' means
        qubit 0 measured in basis state |1>, and qubit 1 & 2 both measured in
        state |0>.

        Args:
            statevector(list or ndarray(complex)): an iterable 1D data-structure
                containing the amplitudes.

        Returns:
            dict: A dictionary whose keys are bitstrings representing the
                multi-qubit states with the least significant qubit first (e.g
                '100' means qubit 0 in state |1>, and qubit 1 and 2 in state
                |0>), and the associated value is the corresponding frequency.
                Unless threshold=0., this dictionary will be sparse.
        """

        n_qubits = int(math.log2(len(statevector)))
        frequencies = dict()
        for i, amplitude in enumerate(statevector):
            frequency = abs(amplitude)**2
            if (frequency - self.freq_threshold) >= 0.:
                frequencies[self.__int_to_binstr(i, n_qubits)] = frequency

        # If n_shots, has been specified, then draw that amount of samples from the distribution
        # and return empirical frequencies instead. Otherwise, return the exact frequencies
        if not self.n_shots:
            return frequencies
        else:
            xk, pk = [], []
            for k, v in frequencies.items():
                xk.append(int(k[::-1], 2))
                pk.append(frequencies[k])
            distr = stats.rv_discrete(name='distr', values=(np.array(xk), np.array(pk)))

            # Generate samples from distribution. Cut in chunks to ensure samples fit in memory, gradually accumulate
            chunk_size = 10**7
            n_chunks = self.n_shots // chunk_size
            freqs_shots = Counter()

            for i in range(n_chunks+1):
                this_chunk = self.n_shots % chunk_size if i == n_chunks else chunk_size
                samples = distr.rvs(size=this_chunk)
                freqs_shots += Counter(samples)
            freqs_shots = {self.__int_to_binstr_lsq(k, n_qubits): v / self.n_shots for k, v in freqs_shots.items()}
            return freqs_shots

    def __int_to_binstr(self, i, n_qubits):
        """Convert an integer into a bit string of size n_qubits, in the order
        specified for the state vector.
        """
        bs = bin(i).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs
        return state_binstr[::-1] if (self.statevector_order == "msq_first") else state_binstr

    def __int_to_binstr_lsq(self, i, n_qubits):
        """Convert an integer into a bit string of size n_qubits, in the
        least-significant qubit order.
        """
        bs = bin(i).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs
        return state_binstr[::-1]
