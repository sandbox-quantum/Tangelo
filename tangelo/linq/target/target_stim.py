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

"""The stim simulation package is made for fast simulation of Clifford circuits.
See https://arxiv.org/abs/2103.02202v3 for more details.
"""

from collections import Counter

import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator.translate_stim import translate_tableau
from tangelo.linq.helpers.circuits.measurement_basis import pauli_of_to_string, measurement_basis_gates


class StimSimulator(Backend):
    """Interface to the stim simulator"""
    def __init__(self, n_shots=None, noise_model=None):
        import stim
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.stim = stim

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None, desired_meas_result=None):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on if return_statevector is set to True.
        The initial_statevector, and desired_meas_result features are currently not implemented
        and will return and error if not None.

        Args:
            source_circuit (Circuit): a circuit in the abstract format to be translated
                for the target backend.
            return_statevector (bool): option to return the statevector
            initial_statevector (list/array) : Not currently implemented, will raise an error
            desired_meas_result (str) : Not currently implemented, will raise an error

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        if initial_statevector is not None:
            raise NotImplementedError("initial_statevector not yet implemented with stim ")
        if desired_meas_result is not None:
            raise NotImplementedError("desired_meas_result not yet implemented with stim ")

        if self.n_shots or self._noise_model:
            translated_circuit = translate_c(source_circuit, "stim",
                 output_options={"noise_model": self._noise_model})
            for qubit in range(source_circuit.width):
                translated_circuit.append("M", [qubit])
            isamples = translated_circuit.compile_sampler().sample(self.n_shots)
            samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]
            frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
        else:
            self._current_state = translate_tableau(source_circuit).state_vector()
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def _get_expectation_value_from_statevector(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        """
        Calculates the expectation value of a qubit operator using a TableauSimulator.

        Args:
            qubit_operator (QubitOperator): The qubit operator for which the expectation value is calculated.
            state_prep_circuit (Circuit): The stabilizer circuit used to prepare the quantum state.
            initial_statevector (list/array) : Not currently implemented, will raise an error
            desired_meas_result (str) : Not currently implemented, will raise an error

        Returns:
            float: The real-valued expectation value of the qubit operator.
        """
        n_qubits = state_prep_circuit.width

        if initial_statevector is not None:
            raise NotImplementedError("initial_statevector not yet implemented with stim ")
        if desired_meas_result is not None:
            raise NotImplementedError("desired_meas_result not yet implemented with stim ")

        if self.n_shots:
            return self._get_expectation_value_from_frequencies(qubit_operator, state_prep_circuit)

        else:
            s = translate_tableau(state_prep_circuit)
            paulisum = 0
            for term, coef in qubit_operator.terms.items():
                if len(term) > n_qubits:  # Cannot have a qubit index beyond circuit size
                    raise ValueError(f"Size of term in qubit operator beyond number of qubits in circuit ({n_qubits}).\n Term = {term}")
                elif not term:  # Empty term: no simulation needed
                    paulisum += coef
                    continue
                paulisum += coef * s.peek_observable_expectation(self.stim.PauliString(pauli_of_to_string(term, n_qubits)))
        return np.real(paulisum)

    def _get_expectation_value_from_frequencies(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        """Take as input a qubit operator H and a state preparation returning a ket |\psi>.
        Return the expectation value <\psi | H | \psi> computed using the frequencies of observable states.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (list/array) : Not currently implemented, will raise an error
            desired_meas_result (str) : Not currently implemented, will raise an error

        Returns:
            complex: The expectation value of this operator with regard to the
                state preparation.
        """
        if initial_statevector is not None:
            raise NotImplementedError("initial_statevector not yet implemented with stim ")
        if desired_meas_result is not None:
            raise NotImplementedError("desired_meas_result not yet implemented with stim ")

        n_qubits = state_prep_circuit.width
        expectation_value = 0.
        for term, coef in qubit_operator.terms.items():
            if len(term) > n_qubits:
                raise ValueError(f"Size of term in qubit operator beyond number of qubits in circuit ({n_qubits}).\n Term = {term}")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue
            basis_circuit = Circuit(measurement_basis_gates(term))
            full_circuit = state_prep_circuit + basis_circuit if (basis_circuit.size > 0) else state_prep_circuit
            frequencies, _ = self.simulate(full_circuit)
            expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
            expectation_value += coef * expectation_term
        return expectation_value

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
