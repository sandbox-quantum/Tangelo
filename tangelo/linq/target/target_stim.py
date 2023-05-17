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
from collections import Counter

import stim
from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator.translate_stim import direct_tableau

class StimSimulator(Backend):
    """Interface to the stim simulator."""

    def __init__(self, n_shots=None, noise_model=None):
        import stim
        from stim import TableauSimulator
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.stim = stim

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
        # only make circuit if doing n>1 shots, otherwise use tableausim, noise model doesn't matter

        # translated_circuit = translate_c(source_circuit, "stim",
        #     output_options={"noise_model": self._noise_model})
        #translated_circuit = direct_tableau(source_circuit)

        #if initial_statevector is not None:
            #initial_state_circuit = self.stim.Tableau.from_state_vector(state_vector=initial_statevector,
                                                                #   endian= 'little').to_circuit(method="elimination")
            #translated_circuit = initial_state_circuit + translated_circuit

        if return_statevector and self.n_shots <= 1:
            self._current_state = direct_tableau(source_circuit).state_vector
        else:
            return_statevector = False

        if self.n_shots > 1:
            translated_circuit = translate_c(source_circuit, "stim",
                 output_options={"noise_model": self._noise_model})
            for qubit in range(source_circuit.width):
                translated_circuit.append("M", [qubit])
            isamples = translated_circuit.compile_sampler().sample(self.n_shots)
            samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]
            frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
        else:
            frequencies = self._statevector_to_frequencies(self._current_state.to_state_vector())

        return (frequencies, np.array(self._current_state.to_state_vector())) if return_statevector else (frequencies, None)

    def expectation_value_from_stabilizer_circuit(self, qubit_operator, state_prep_circuit, n_qubits=None):
        from tangelo.linq.helpers.circuits.measurement_basis import pauli_of_to_string
        if not n_qubits:
            n_qubits = state_prep_circuit.width
        s = direct_tableau(state_prep_circuit, self._noise_model)
        paulisum = 0
        for term, coef in qubit_operator.terms.items():
            paulisum += coef * s.peek_observable_expectation(self.stim.PauliString(pauli_of_to_string(term, n_qubits)))
        return np.real(paulisum)

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}