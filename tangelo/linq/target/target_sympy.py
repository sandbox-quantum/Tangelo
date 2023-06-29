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
        super().__init__(n_shots, noise_model)

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
        """This simulator manipulates symbolic expressions, i.e. gates can have
        unspecified parameters (strings interpreted as variables). As with the
        other simulators, it performs state preparation corresponding to the
        input circuit, returns the frequencies of the different observables, and
        either the statevector or None depending on if return_statevector is set
        to True.

        Args:
            source_circuit (Circuit): A circuit in the abstract format to be
                translated for the target backend.
            return_statevector (bool): Option to return the statevector as well,
                if available.
            initial_statevector (array/matrix or sympy.physics.quantum.Qubit): A
                valid statevector in the format supported by the target backend.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            sympy.Matrix: The symbolic statevector, if requested
                by the user (if not, set to None).
        """

        from sympy import simplify
        from sympy.physics.quantum import qapply
        from sympy.physics.quantum.qubit import Qubit, matrix_to_qubit, \
            qubit_to_matrix, measure_all

        translated_circuit = translate_c(source_circuit, "sympy")

        # Transform the initial_statevector if it is provided.
        if initial_statevector is None:
            python_statevector = Qubit("0"*(source_circuit.width))
        elif isinstance(initial_statevector, Qubit):
            python_statevector = initial_statevector
        elif isinstance(initial_statevector, (np.ndarray, np.matrix)):
            python_statevector = matrix_to_qubit(initial_statevector)
        else:
            raise ValueError(f"The {type(initial_statevector)} type for initial_statevector is not supported.")

        # Deterministic circuit, run once.
        state = qapply(translated_circuit * python_statevector)
        self._current_state = state
        python_statevector = qubit_to_matrix(state)

        measurements = measure_all(state)

        frequencies = dict()
        for vec, prob in measurements:
            prob = simplify(prob, tolerance=1e-4).evalf()
            if not prob.is_zero:
                bistring = "".join(str(bit) for bit in reversed(vec.qubit_values))
                frequencies[bistring] = prob

        return (frequencies, python_statevector) if return_statevector else (frequencies, None)

    def expectation_value_from_prepared_state(self, qubit_operator, n_qubits, prepared_state=None):
        """Compute an expectation value using a representation of the state
        using sympy functionalities.

        Args:
            qubit_operator (QubitOperator): a qubit operator in tangelo format
            n_qubits (int): Number of qubits.
            prepared_state (array/matrix or sympy.physics.quantum.Qubit): A
                numpy or a sympy object representing the state. Internally, a
                numpy object is transformed into the sympy representation.
                Default is None, in this case it is set to the current state in
                the simulator object.

        Returns:
            sympy.core.add.Add: Eigenvalue represented as a symbolic sum.
        """

        from sympy import simplify, cos
        from sympy.physics.quantum import Dagger

        prepared_state = self._current_state if prepared_state is None else prepared_state
        operator = translate_operator(qubit_operator, source="tangelo", target="sympy", n_qubits=n_qubits)

        eigenvalue = Dagger(prepared_state) * operator * prepared_state
        eigenvalue = eigenvalue[0, 0].rewrite(cos)
        eigenvalue = simplify(eigenvalue).evalf()

        return eigenvalue

    @staticmethod
    def backend_info():
        return {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": False}
