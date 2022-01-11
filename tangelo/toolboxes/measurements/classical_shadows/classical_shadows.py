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

"""This file provides an API enabling the use of classical shadows. The original
idea is described in H.Y. Huang, R. Kueng, and J. Preskill, Nature Physics 16,
1050 (2020).
"""

import abc
import warnings

import numpy as np

from tangelo.linq.circuit import Circuit


# State |0> or |1>.
zero_state = np.array([1, 0])
one_state = np.array([0, 1])

# Pauli matrices.
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]])
matrices = {"X": X, "Y": Y, "Z": Z}

# Traces of each Pauli matrices.
traces = {pauli: np.trace(matrix) for pauli, matrix in matrices.items()}

# Reverse channels to undo single Pauli rotations.
S_T = np.array([[1, 0], [0, -1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I = np.array([[1, 0], [0, 1]])
rotations = {"X": H, "Y": H @ S_T, "Z": I}


class ClassicalShadow(abc.ABC):
    """Abstract class for the classical shadows implementation. Classical
    shadows is a mean to characterize a quantum state (within an error treshold)
    with the fewest measurement possible.
    """

    def __init__(self, circuit, bitstrings=None, unitaries=None):
        """Default constructor for the ClassicalShadow object. This class is
        the parent class for the different classical shadows flavors. The object
        is defined by the bistrings and unitaries used in the process. Abstract
        methods are defined to take into account the procedure to inverse the
        channel.

        Args:
            bistrings (list of str): Representation of the outcomes for all
                snapshots. E.g. ["11011", "10000", ...].
            unitaries (list of str): Representation of the unitary for every
                snapshot, used to reverse the channel.
        """

        self.circuit = circuit
        self.bitstrings = list() if bitstrings is None else bitstrings
        self.unitaries = list() if unitaries is None else unitaries

        # If the state has been estimated, it is stored into this attribute.
        self.state_estimate = None

    @property
    def n_qubits(self):
        """Returns the number of qubits the shadow represents."""
        return self.circuit.width

    @property
    def size(self):
        """Number of shots used to make the shadow."""
        return len(self.bitstrings)

    @property
    def unique_unitaries(self):
        """Returns the list of unique unitaries."""
        return list(set(self.unitaries))

    def __len__(self):
        """Same as the shadow size."""
        return self.size

    def append(self, bitstring, unitary):
        """Append method to merge new snapshots to an existing shadow.

        Args:
            bistring (str or list of str): Representation of outcomes.
            unitary (str or list of str): Relevant unitary for those outcomes.
        """
        if isinstance(bitstring, list) and isinstance(unitary, list):
            assert len(bitstring) == len(unitary)
            self.bitstrings += bitstring
            self.unitaries += unitary
        elif isinstance(bitstring, str) and isinstance(unitary, str):
            self.bitstrings.append(bitstring)
            self.unitaries.append(unitary)
        else:
            raise ValueError("bistring and unitary arguments must be consistent strings or list of strings.")

    def get_observable(self, qubit_op, *args, **kwargs):
        """Getting an estimated observable value for a qubit operator from the
        classical shadow. This function loops through all terms and calls, for
        each of them, the get_term_observable method defined in the child class.
        Other arguments (args, kwargs) can be passed to the method.

        Args:
            qubit_op (QubitOperator): Operator to estimate.
        """
        observable = 0.
        for term, coeff in qubit_op.terms.items():
            observable += self.get_term_observable(term, coeff, *args, **kwargs)

        return observable

    def simulate(self, backend, initial_statevector=None):
        """Simulate, using a predefined backend, a shadow from a circuit or a
        statevector.

        Args:
            backend (Simulator): Backend for the simulation of a shadow.
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend.
        """

        if not self.unitaries:
            raise ValueError(f"The build method of {self.__class__.__name__} must be called before simulation.")

        if backend.n_shots != 1:
            warnings.warn(f"Changing number of shots to 1 for the backend (classical shadows).")
            backend.n_shots = 1

        # Different behavior if circuit or initial_statevector is defined.
        one_shot_circuit_template = self.circuit if self.circuit is not None else Circuit(n_qubits=self.n_qubits)

        for basis_circuit in self.get_basis_circuits(only_unique=False):
            one_shot_circuit = one_shot_circuit_template + basis_circuit if (basis_circuit.size > 0) else one_shot_circuit_template

            # Frequencies returned by simulate are of the form {'0100...01': 1.0}.
            # We add the bitstring to the shadow.
            freqs, _ = backend.simulate(one_shot_circuit, initial_statevector=initial_statevector)
            self.bitstrings += [list(freqs.keys())[0]]

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def get_basis_circuits(self, only_unique=False):
        pass

    @abc.abstractmethod
    def get_term_observable(self):
        pass
