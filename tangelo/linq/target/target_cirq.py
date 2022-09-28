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

from collections import Counter

import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.linq.simulator import SimulatorBase
import tangelo.linq.translator as translator
from tangelo.toolboxes.post_processing.histogram import Histogram


class Cirq(SimulatorBase):

    def __init__(self, n_shots=None, noise_model=None):
        """Instantiate cirq simulator object.

        Args:
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format
                expected from the target backend.
        """
        super().__init__(n_shots=n_shots, noise_model=noise_model, backend_info=self.backend_info)

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
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

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        import cirq

        translated_circuit = translator.translate_cirq(source_circuit, self._noise_model)

        if source_circuit.is_mixed_state or self._noise_model:
            # Only DensityMatrixSimulator handles noise well, can use Simulator but it is slower
            cirq_simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
        else:
            cirq_simulator = cirq.Simulator(dtype=np.complex128)

        # If requested, set initial state
        cirq_initial_statevector = initial_statevector if initial_statevector is not None else 0

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state:
            # cirq.dephase_measurements changes measurement gates to Krauss operators so simulators
            # can be called once and density matrix sampled repeatedly.
            translated_circuit = cirq.dephase_measurements(translated_circuit)
            sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = sim.final_density_matrix
            indices = list(range(source_circuit.width))
            isamples = cirq.sample_density_matrix(sim.final_density_matrix, indices, repetitions=self.n_shots)
            samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]

            frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
        # Noiseless simulation using the statevector simulator otherwise
        else:
            job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = job_sim.final_state_vector
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):
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

    @property
    def backend_info(self):
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}


class QSimCirq(SimulatorBase):

    def __init__(self, n_shots=None, noise_model=None, options=None):
        """
        Instantiate qsimcirq simulator object.

        Args:
            Args:
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model (NoiseModel): A noise model object assumed to be in the format
                expected from the target backend.
            options (dict): The options to use as defined in https://github.com/quantumlib/qsim/blob/master/docs/usage.md
                most important are {'t': num_threads, 'f': max_fused_size} where num_threads and max_fused_size are int.
        """
        super().__init__(n_shots=n_shots, noise_model=noise_model, backend_info=self.backend_info)
        self.options = options if options is not None else dict()

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
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

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        import qsimcirq
        import cirq

        cirq_simulator = qsimcirq.QSimSimulator(self.options)

        # If requested, set initial state
        cirq_initial_statevector = np.array(initial_statevector, dtype=np.complex64) if initial_statevector is not None else None

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state:
            if cirq_initial_statevector is not None:
                sv = StateVector(cirq_initial_statevector, order="lsq_first")
                initial_circuit = sv.initializing_circuit()
            else:
                initial_circuit = Circuit()
            n_meas = source_circuit._gate_counts.get("MEASURE", 0)

            translated_circuit = translator.translate_cirq(initial_circuit) + translator.translate_cirq(source_circuit,
                                                                                                        self._noise_model,
                                                                                                        save_measurements=True)
            qubit_list = cirq.LineQubit.range(source_circuit.width)
            for i, qubit in enumerate(qubit_list):
                translated_circuit.append(cirq.measure(qubit, key=str(i+n_meas)))

            sim = cirq_simulator.run(translated_circuit, repetitions=self.n_shots)
            samples = list()
            for j in range(self.n_shots):
                samples += ["".join([str(sim.measurements[str(i)][j, 0]) for i in range(n_meas+source_circuit.width)])]
            self.all_frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}

            histogram = Histogram(self.all_frequencies, n_shots=self.n_shots, msq_first=False)
            histogram.remove_qubit_indices(list(range(n_meas)))
            frequencies = histogram.frequencies

        # Noiseless simulation using the statevector simulator otherwise
        else:
            translated_circuit = translator.translate_cirq(source_circuit, self._noise_model)
            job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = job_sim.state_vector()
            self._current_state /= np.linalg.norm(self._current_state)
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):
        import cirq

        GATE_CIRQ = translator.get_cirq_gates()
        qubit_labels = cirq.LineQubit.range(n_qubits)
        qubit_map = {q: i for i, q in enumerate(qubit_labels)}
        paulisum = 0.*cirq.PauliString(cirq.I(qubit_labels[0]))
        for term, coef in qubit_operator.terms.items():
            pauli_list = [GATE_CIRQ[pauli](qubit_labels[index]) for index, pauli in term]
            paulisum += cirq.PauliString(pauli_list, coefficient=coef)
        exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
        return np.real(exp_value)

    @property
    def backend_info(self):
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}


class Cirq_QVM(SimulatorBase):

    def __init__(self, n_shots, noise_model, qubits_to_use):
        """Instantiate google QVM object.

        Args:
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model (str): A device identifier.
            qubits_to_use (int): The device qubits to utilize for the calculation.
        """
        super().__init__(n_shots=n_shots, noise_model=noise_model, backend_info=self.backend_info)
        import qsimcirq
        import cirq_google
        self.qubits_to_use = qubits_to_use

        # Construct a simulator with a noise model based on the specified processor.
        cal = cirq_google.engine.load_median_device_calibration(self._noise_model)
        noise_props = cirq_google.noise_properties_from_calibration(cal)
        noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
        sim = qsimcirq.QSimSimulator(noise=noise_model)

        # Create a device from the public device description
        device = cirq_google.engine.create_device_from_processor_id(self._noise_model)
        # Build the simulated local processor from the simulator and device.
        sim_processor = cirq_google.engine.SimulatedLocalProcessor(
            processor_id=self._noise_model, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal}
            )
        # Package the processor to use an Engine interface
        sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])
        sim_device = sim_engine.get_processor(self._noise_model).get_device()
        self.sim_device = sim_device
        self.sim_engine = sim_engine

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
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
            return_statevector(bool): Must be set to false
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        import cirq

        if return_statevector:
            raise ValueError("Return of statevector not available for Cirq_QVM.")
        # If requested, set initial state
        cirq_initial_statevector = np.array(initial_statevector, dtype=np.complex64) if initial_statevector is not None else None

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        if cirq_initial_statevector is not None:
            sv = StateVector(cirq_initial_statevector, order="lsq_first")
            initial_circuit = sv.initializing_circuit()
        else:
            initial_circuit = Circuit()
        n_meas = source_circuit._gate_counts.get("MEASURE", 0)

        translated_circuit = translator.translate_cirq(initial_circuit) + translator.translate_cirq(source_circuit, save_measurements=True)
        qubit_list = cirq.LineQubit.range(source_circuit.width)
        for i, qubit in enumerate(qubit_list):
            translated_circuit.append(cirq.measure(qubit, key=str(i+n_meas)))

        hardware_circuit = cirq.optimize_for_target_gateset(
            translated_circuit, context=cirq.TransformerContext(deep=True), gateset=cirq.SqrtIswapTargetGateset()
            )

        device_qubits = self.qubits_to_use
        if self.qubits_to_use is None:
            raise ValueError("qubits to use must be initialized to use cirq QVM")
        qubit_map = dict(zip(qubit_list, device_qubits[:len(qubit_list)]))
        device_ready_circuit = hardware_circuit.transform_qubits(lambda q: qubit_map[q])

        sim = self.sim_engine.get_sampler(self._noise_model).run(device_ready_circuit, repetitions=self.n_shots)
        samples = list()
        for j in range(self.n_shots):
            samples += ["".join([str(sim.measurements[str(i)][j, 0]) for i in range(n_meas+source_circuit.width)])]
        self.all_frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}

        histogram = Histogram(self.all_frequencies, n_shots=self.n_shots, msq_first=False)
        histogram.remove_qubit_indices(list(range(n_meas)))
        frequencies = histogram.frequencies

        return (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):
        import cirq

        GATE_CIRQ = translator.get_cirq_gates()
        qubit_labels = cirq.LineQubit.range(n_qubits)
        qubit_map = {q: i for i, q in enumerate(qubit_labels)}
        paulisum = 0.*cirq.PauliString(cirq.I(qubit_labels[0]))
        for term, coef in qubit_operator.terms.items():
            pauli_list = [GATE_CIRQ[pauli](qubit_labels[index]) for index, pauli in term]
            paulisum += cirq.PauliString(pauli_list, coefficient=coef)
        exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
        return np.real(exp_value)

    @property
    def backend_info(self):
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}
