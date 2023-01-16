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


import os
from collections import Counter

import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator import translate_operator


class QulacsSimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        import qulacs
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.qulacs = qulacs

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None, save_mid_circuit_meas=False):
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
        translated_circuit = translate_c(source_circuit, "qulacs",
                output_options={"noise_model": self._noise_model, "save_measurements": save_mid_circuit_meas})

        # Initialize state on GPU if available and desired. Default to CPU otherwise.
        if ('QuantumStateGpu' in dir(self.qulacs)) and (int(os.getenv("QULACS_USE_GPU", 0)) != 0):
            state = self.qulacs.QuantumStateGpu(source_circuit.width)
        else:
            state = self.qulacs.QuantumState(source_circuit.width)

        python_statevector = None
        if initial_statevector is not None:
            state.load(initial_statevector)

        # If you don't want to save the mid-circuit measurements for a mixed state
        if (source_circuit.is_mixed_state or self._noise_model) and not save_mid_circuit_meas:
            samples = dict()
            for _ in range(self.n_shots):
                translated_circuit.update_quantum_state(state)
                bitstr = state.sampling(1)[0]
                samples[bitstr] = samples.get(bitstr, 0) + 1
                if initial_statevector is not None:
                    state.load(initial_statevector)
                else:
                    state.set_zero_state()

        # To save mid-circuit measurement results
        elif save_mid_circuit_meas:
            n_meas = source_circuit.counts.get("MEASURE", 0)
            samples = dict()
            for _ in range(self.n_shots):
                translated_circuit.update_quantum_state(state)
                measurement = "".join([str(state.get_classical_value(i)) for i in range(n_meas)])
                sample = self._int_to_binstr(state.sampling(1)[0], source_circuit.width)
                bitstr = measurement + sample
                samples[bitstr] = samples.get(bitstr, 0) + 1
                if initial_statevector is not None:
                    state.load(initial_statevector)
                else:
                    state.set_zero_state()
            self.all_frequencies = {k: v / self.n_shots for k, v in samples.items()}
            return (self.all_frequencies, python_statevector) if return_statevector else (self.all_frequencies, None)

        # All other cases for shot-based simulation
        elif self.n_shots is not None:
            translated_circuit.update_quantum_state(state)
            self._current_state = state
            python_statevector = np.array(state.get_vector()) if return_statevector else None
            samples = Counter(state.sampling(self.n_shots))  # this sampling still returns a list

        # Statevector simulation
        else:
            translated_circuit.update_quantum_state(state)
            self._current_state = state
            python_statevector = np.array(state.get_vector())
            frequencies = self._statevector_to_frequencies(python_statevector)
            return (frequencies, python_statevector) if return_statevector else (frequencies, None)

        frequencies = {self._int_to_binstr(k, source_circuit.width): v / self.n_shots
                       for k, v in samples.items()}
        return (frequencies, python_statevector) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, prepared_state=None):
        """ Compute an expectation value using a representation of the state
        using qulacs functionalities.

        Args:
            qubit_operator (QubitOperator): a qubit operator in tangelo format
            prepared_state (np.array): a numpy array encoding the state (can be
                a vector or a matrix). Default is None, in this case it is set
                to the current state in the simulator object.

        Returns:
            float64 : the expectation value of the qubit operator w.r.t the input state
        """
        if prepared_state is None:
            prepared_state = self._current_state

        operator = translate_operator(qubit_operator, source="tangelo", target="qulacs")
        return operator.get_expectation_value(prepared_state).real

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
