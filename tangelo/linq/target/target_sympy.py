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


import numpy as np

from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c
from tangelo.linq.translator import translate_operator


class SympySimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        import sympy
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.sympy = sympy
        self.quantum = sympy.physics.quantum

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For this
        simulator, symbolic expression are supported. Gates can have unspecified
        parameters.

        Args:
            source_circuit (Circuit): A circuit in the abstract format to be
                translated for the target backend.
            return_statevector (bool): Option to return the statevector as well,
                if available.
            initial_statevector (array/matrix or sympy.physics.quantum.Qubit): A v
                alid statevector in the format supported by the target backend.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            sympy.Matrix: The symbolic statevector, if requested
                by the user (if not, set to None).
        """

        translated_circuit = translate_c(source_circuit, "sympy")

        # Transform the initial_statevector if it is provided.
        if initial_statevector is None:
            python_statevector = self.quantum.qubit.Qubit("0"*(source_circuit.width))
        elif isinstance(initial_statevector, self.quantum.qubit.Qubit):
            python_statevector = initial_statevector
        elif isinstance(initial_statevector, (np.ndarray, np.matrix)):
            python_statevector = self.quantum.qubit.matrix_to_qubit(initial_statevector)
        else:
            raise ValueError(f"The {type(initial_statevector)} type for initial_statevector is not supported.")

        # Deterministic circuit, run once.
        state = self.quantum.qapply(translated_circuit * python_statevector)
        self._current_state = state
        python_statevector = self.quantum.qubit.qubit_to_matrix(state)

        measurements = self.quantum.qubit.measure_all(state)
        frequencies = {"".join(str(bit) for bit in reversed(vec.qubit_values)): self.sympy.simplify(prob)
            for vec, prob in measurements}

        return (frequencies, python_statevector) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state=None):
        """Compute an expectation value using a representation of the state
        using sympy functionalities.

        Args:
            qubit_operator (QubitOperator): a qubit operator in tangelo format
            n_qubits (int): Number of qubits.
            prepared_state (array/matrix or sympy.physics.quantum.Qubit): A
                numpy or a sympy object encapsulating the state. Internally, a
                numpy object is transformed into the sympy representation.
                Default is None, in this case it is set to the current state in
                the simulator object.

        Returns:
            sympy.core.add.Add: Eigenvalue represented as a symnbolic sum.
        """

        prepared_state = self._current_state if prepared_state is None else prepared_state
        operator = translate_operator(qubit_operator, source="tangelo", target="sympy", n_qubits=n_qubits)
        eigenvalue = self.quantum.qapply(self.quantum.Dagger(prepared_state) * operator * prepared_state)

        return eigenvalue

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": False}
