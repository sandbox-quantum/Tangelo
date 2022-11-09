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

        translated_circuit = translate_c(source_circuit, "qiskit")

        # If requested, set initial state
        if initial_statevector is not None:
            if self._noise_model:
                raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")
            else:
                n_qubits = int(math.log2(len(initial_statevector)))
                initial_state_circuit = self.qiskit.QuantumCircuit(n_qubits, n_qubits)
                initial_state_circuit.initialize(initial_statevector, list(range(n_qubits)))
                translated_circuit = initial_state_circuit.compose(translated_circuit)

        # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state:
            from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model

            meas_range = range(source_circuit.width)
            translated_circuit.measure(meas_range, meas_range)
            return_statevector = False
            backend = self.AerSimulator()

            qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
            opt_level = 0 if self._noise_model else None

            job_sim = self.qiskit.execute(translated_circuit, backend, noise_model=qiskit_noise_model,
                                          shots=self.n_shots, basis_gates=None, optimization_level=opt_level)
            sim_results = job_sim.result()
            frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

        # Noiseless simulation using the statevector simulator otherwise
        else:
            backend = self.AerSimulator(method='statevector')
            translated_circuit = self.qiskit.transpile(translated_circuit, backend)
            translated_circuit.save_statevector()
            sim_results = backend.run(translated_circuit).result()
            self._current_state = np.asarray(sim_results.get_statevector(translated_circuit))
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(sim_results.get_statevector())) if return_statevector else (frequencies, None)

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
