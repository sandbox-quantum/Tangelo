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

import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.simulator import SimulatorBase
import tangelo.linq.translator as translator
from tangelo.toolboxes.post_processing.histogram import Histogram


class Qiskit(SimulatorBase):
    """Interface to the qiskit simulator."""

    def __init__(self, n_shots=None, noise_model=None):
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
        import qiskit
        from qiskit_aer import AerSimulator

        translated_circuit = translator.translate_qiskit(source_circuit, save_measurements=True)

        # If requested, set initial state
        if initial_statevector is not None:
            if self._noise_model:
                raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")
            else:
                n_qubits = int(math.log2(len(initial_statevector)))
                initial_state_circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
                initial_state_circuit.initialize(initial_statevector, list(range(n_qubits)))
                translated_circuit = initial_state_circuit.compose(translated_circuit)

        # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state:
            from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model
            n_meas = source_circuit._gate_counts.get("MEASURE", 0)
            meas_range = range(n_meas, n_meas + source_circuit.width)
            translated_circuit.measure(range(source_circuit.width), meas_range)
            return_statevector = False
            backend = AerSimulator()
            qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
            opt_level = 0 if self._noise_model else None

            job_sim = qiskit.execute(translated_circuit, backend, noise_model=qiskit_noise_model,
                                     shots=self.n_shots, basis_gates=None, optimization_level=opt_level)
            sim_results = job_sim.result()
            self.all_frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

            self.histogram = Histogram(self.all_frequencies, n_shots=self.n_shots, msq_first=False)
            self.histogram.remove_qubit_indices(*list(range(n_meas)))
            frequencies = self.histogram.frequencies

        # Noiseless simulation using the statevector simulator otherwise
        else:
            backend = AerSimulator(method='statevector')
            translated_circuit = qiskit.transpile(translated_circuit, backend)
            translated_circuit.save_statevector()
            sim_results = backend.run(translated_circuit).result()
            self._current_state = np.asarray(sim_results.get_statevector(translated_circuit))
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(sim_results.get_statevector())) if return_statevector else (frequencies, None)

    @property
    def backend_info(self):
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}


class QiskitDevice(SimulatorBase):
    """Interface to using FakeDevices from qiskit."""

    def __init__(self, n_shots=None, noise_model=None, qubits_to_use=None, opt_level=1, meas_mitt=False):
        super().__init__(n_shots=n_shots, noise_model=noise_model, backend_info=self.backend_info)
        self.qubits_to_use = qubits_to_use
        self.opt_level = opt_level
        self.meas_mitt = meas_mitt

    def translate(self, source_circuit, qubits_to_use):
        """Take in an abstract circuit, return an equivalent qiskit QuantumCircuit
        instance
        Args:
            source_circuit: quantum circuit in the abstract format.
            qubits_to_use: list
            return_registers (bool): whether to return the registers to simulate
        Returns:
            qiskit.QuantumCircuit: the corresponding qiskit quantum circuit. if return_registers=False
            (qiskit.QuantumCircuit, qiskit.QuantumRegister, qiskit.ClassicalRegister) if return_registers=True
        """

        import qiskit

        GATE_QISKIT = translator.get_qiskit_gates()
        num_virtual_qubits = len(source_circuit._qubit_indices) if qubits_to_use is not None else source_circuit.width
        q = qiskit.QuantumRegister(num_virtual_qubits, name="q")
        c = qiskit.ClassicalRegister(num_virtual_qubits, name="c")
        target_circuit = qiskit.QuantumCircuit(q, c)

        # Maps the gate information properly. Different for each backend (order, values)
        for gate in source_circuit._gates:
            if gate.control is not None:
                if len(gate.control) > 1:
                    raise ValueError('Multi-controlled gates not supported with qiskit. Gate {gate.name} with controls {gate.control} is not allowed')
            if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
                (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]])
            elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
                (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.target[0]])
            elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
                (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.control[0]], q[gate.target[0]])
            elif gate.name in {"CNOT", "CH", "CX", "CY", "CZ"}:
                (GATE_QISKIT[gate.name])(target_circuit, q[gate.control[0]], q[gate.target[0]])
            elif gate.name in {"SWAP"}:
                (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]], q[gate.target[1]])
            elif gate.name in {"CSWAP"}:
                (GATE_QISKIT[gate.name])(target_circuit, q[gate.control[0]], q[gate.target[0]], q[gate.target[1]])
            elif gate.name in {"XX"}:
                (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.target[0]], q[gate.target[1]])
            elif gate.name in {"MEASURE"}:
                (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]], c[gate.target[0]])
            else:
                raise ValueError(f"Gate '{gate.name}' not supported on backend qiskit")
        return target_circuit, q, c

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
        import qiskit
        from qiskit.providers.fake_provider import fake_provider
        from qiskit_aer import AerSimulator
        from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter

        translated_circuit, q, c = self.translate(source_circuit, self.qubits_to_use)

        # If requested, set initial state
        if initial_statevector is not None:
            raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")

        # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
        num_measure_qubits = len(self.qubits_to_use) if self.qubits_to_use is not None else source_circuit.width
        translated_circuit.measure(q, c)

        virtual_to_physical = dict()
        qubit_map = self.qubits_to_use if self.qubits_to_use is not None else [i for i in range(num_measure_qubits)]
        if len(qubit_map) != num_measure_qubits:
            raise ValueError("number of qubits_to_use must equal number of qubits in circuit")
        for i in range(num_measure_qubits):
            virtual_to_physical[q[i]] = qubit_map[i]

        return_statevector = False
        if self._noise_model is not None:
            try:
                device_to_call = getattr(fake_provider, self._noise_model)
            except AttributeError:
                raise ValueError(f"{self._noise_model} is not one of the Fake Qiskit backends")
        else:
            raise ValueError("_device_name must be included in a noise_model to run a simulated device")
        device_backend = device_to_call()
        backend = AerSimulator.from_backend(device_backend)

        if self.meas_mitt:
            meas_calibs, state_labels = complete_meas_cal(qr=q, circlabel='mcal')
            t_qc = qiskit.transpile(meas_calibs, backend, initial_layout=virtual_to_physical)
            qobj = qiskit.assemble(t_qc, shots=10000)
            cal_results = backend.run(qobj, shots=10000).result()
            meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
            meas_filter = meas_fitter.filter

        job_sim = qiskit.execute(qiskit.transpile(translated_circuit, backend, initial_layout=virtual_to_physical,
                                 optimization_level=self.opt_level), backend, shots=self.n_shots, basis_gates=None)
        if self.meas_mitt:
            presim_results = job_sim.result()
            sim_results = meas_filter.apply(presim_results)
        else:
            sim_results = job_sim.result()

        frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

        return (frequencies, np.array(sim_results.get_statevector())) if return_statevector else (frequencies, None)

    @property
    def backend_info(self):
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
