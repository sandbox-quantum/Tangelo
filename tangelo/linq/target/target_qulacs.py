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


class QulacsSimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        import qulacs
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.qulacs = qulacs

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

        translated_circuit = translate_c(source_circuit, "qulacs",
            output_options={"noise_model": self._noise_model})

        # Initialize state on GPU if available and desired. Default to CPU otherwise.
        if ('QuantumStateGpu' in dir(self.qulacs)) and (int(os.getenv("QULACS_USE_GPU", 0)) != 0):
            state = self.qulacs.QuantumStateGpu(source_circuit.width)
        else:
            state = self.qulacs.QuantumState(source_circuit.width)
        if initial_statevector is not None:
            state.load(initial_statevector)

        if (source_circuit.is_mixed_state or self._noise_model):
            samples = list()
            for i in range(self.n_shots):
                translated_circuit.update_quantum_state(state)
                samples.append(state.sampling(1)[0])
                if initial_statevector is not None:
                    state.load(initial_statevector)
                else:
                    state.set_zero_state()
            python_statevector = None
        elif self.n_shots is not None:
            translated_circuit.update_quantum_state(state)
            self._current_state = state
            python_statevector = np.array(state.get_vector()) if return_statevector else None
            samples = state.sampling(self.n_shots)
        else:
            translated_circuit.update_quantum_state(state)
            self._current_state = state
            python_statevector = state.get_vector()
            frequencies = self._statevector_to_frequencies(python_statevector)
            return (frequencies, np.array(python_statevector)) if return_statevector else (frequencies, None)

        frequencies = {self._int_to_binstr(k, source_circuit.width): v / self.n_shots
                       for k, v in Counter(samples).items()}
        return (frequencies, python_statevector)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):

        # TODO: This section previously used qulacs.quantum_operator.create_quantum_operator_from_openfermion_text but was changed
        # due to a memory leak. We can re-evaluate the implementation if/when Issue #303 (https://github.com/qulacs/qulacs/issues/303)
        # is fixed.
        operator = self.qulacs.Observable(n_qubits)
        for term, coef in qubit_operator.terms.items():
            pauli_string = "".join(f" {op} {qu}" for qu, op in term)
            operator.add_operator(coef, pauli_string)
        return operator.get_expectation_value(self._current_state).real

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
