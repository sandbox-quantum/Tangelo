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
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator import get_cirq_gates


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

        translated_circuit = translate_c(source_circuit, "cirq",
            output_options={"noise_model": self._noise_model})

        if source_circuit.is_mixed_state or self._noise_model:
            # Only DensityMatrixSimulator handles noise well, can use Simulator but it is slower
            cirq_simulator = self.cirq.DensityMatrixSimulator(dtype=np.complex128)
        else:
            cirq_simulator = self.cirq.Simulator(dtype=np.complex128)

        # If requested, set initial state
        cirq_initial_statevector = initial_statevector if initial_statevector is not None else 0

        # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
        if self._noise_model or source_circuit.is_mixed_state:
            # cirq.dephase_measurements changes measurement gates to Krauss operators so simulators
            # can be called once and density matrix sampled repeatedly.
            translated_circuit = self.cirq.dephase_measurements(translated_circuit)
            sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = sim.final_density_matrix
            indices = list(range(source_circuit.width))
            isamples = self.cirq.sample_density_matrix(sim.final_density_matrix, indices, repetitions=self.n_shots)
            samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]

            frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
        # Noiseless simulation using the statevector simulator otherwise
        else:
            job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
            self._current_state = job_sim.final_state_vector
            frequencies = self._statevector_to_frequencies(self._current_state)

        return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state):

        GATE_CIRQ = get_cirq_gates()
        qubit_labels = self.cirq.LineQubit.range(n_qubits)
        qubit_map = {q: i for i, q in enumerate(qubit_labels)}
        paulisum = 0.*self.cirq.PauliString(self.cirq.I(qubit_labels[0]))
        for term, coef in qubit_operator.terms.items():
            pauli_list = [GATE_CIRQ[pauli](qubit_labels[index]) for index, pauli in term]
            paulisum += self.cirq.PauliString(pauli_list, coefficient=coef)
        if self._noise_model:
            exp_value = paulisum.expectation_from_density_matrix(prepared_state, qubit_map)
        else:
            exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
        return np.real(exp_value)

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}
