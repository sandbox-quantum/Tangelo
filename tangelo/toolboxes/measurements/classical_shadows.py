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
import random
import warnings

import numpy as np

from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of


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
    def estimate_state(self):
        pass

    @abc.abstractmethod
    def get_term_observable(self):
        pass


class RandomizedClassicalShadow(ClassicalShadow):
    r"""Classical shadows using randomized single Pauli measurements, as defined
    in H.Y. Huang, R. Kueng, and J. Preskill, Nature Physics 16, 1050 (2020). In
    short, the channel is inversed to geet the state with the formula
    \hat{\rho} = \bigotimes_{j=1}^n \left( 3U_j^{\dagger} |b_j\rangle \langle b_j| U_j - \mathbb{I} \right)
    """

    def build(self, n_shots):
        """Random sampling of single pauli words.

        Args:
            n_shots (int): Total number of measurements.

        Returns:
            list of str: Measurements generated for a randomized procedure.
        """
        measurement_procedure = []
        for _ in range(n_shots):
            single_round_measurement = "".join([random.choice(["X", "Y", "Z"]) for _ in range(self.n_qubits)])
            measurement_procedure.append(single_round_measurement)

        self.unitaries = measurement_procedure
        return measurement_procedure

    def get_basis_circuits(self, only_unique=False):
        """Output a list of circuits corresponding to the random Pauli words
        unitaries.

        Args:
            only_unique (bool): Considering only unique unitaries.

        Returns:
            list of Circuit or tuple: All basis circuits or a tuple of unique
                circuits (first) with the numbers of occurence (last).
        """

        if not self.unitaries:
            raise ValueError(f"A set of unitaries must de defined (can be done with the build method in {self.__class__.__name__}).")

        unitaries_to_convert = self.unique_unitaries if only_unique else self.unitaries

        basis_circuits = list()
        for pauli_word in unitaries_to_convert:
            # Transformation of a unitary to quantum gates.
            pauli_of = pauli_string_to_of(pauli_word)
            basis_circuits += [Circuit(measurement_basis_gates(pauli_of), self.n_qubits)]

        # Counting each unique circuits (use for reversing to a full shadow from an experiement on hardware).
        if only_unique:
            unique_basis_circuits = [(basis_circuits[i], self.unitaries.count(u)) for i, u in enumerate(unitaries_to_convert)]
            return unique_basis_circuits
        # Not necessary if we output everything.
        else:
            return basis_circuits

    def estimate_state(self, start=0, end=None, indices=None):
        """Returns the classical shadow average density matrix for a range of
        snapshots.

        Args:
            start (int): Starting snapshot for the desired range.
            end (int): Ending snapshot for the desired range.
            indices (list int): Specific snapshot to pick. If this
                variable is set, start and end are ignored.

        Returns:
            array of complex: Estimation of the 2^n * 2^n state.
        """

        # Select specific snapshots. Default: all snapshots.
        if indices is not None:
            snapshot_indices = indices
        else:
            if end is None:
                end = self.size
            snapshot_indices = list(range(start, min(end, self.size)))

        # Creation of the density matrix object that snapshot rho will be added to.
        rho = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)

        # Undo rotations for the selected snapshot(s).
        for i_snapshot in snapshot_indices:

            # Starting point is the Identity matrix of size 1.
            rho_snapshot = np.ones((1, 1), dtype=complex)

            for n in range(self.n_qubits):
                state = zero_state if self.bitstrings[i_snapshot][n] == "0" else one_state

                # Unitary to undo the rotation.
                U = rotations[self.unitaries[i_snapshot][n]]
                bU = state @ U
                rho_i = 3 * np.outer(bU.conj(), bU) - I
                rho_snapshot = np.kron(rho_snapshot, rho_i)

            rho += rho_snapshot[:, :]

        # Save the result, return the state estimation
        self.state_estimate = rho / len(snapshot_indices)
        return self.state_estimate

    def get_term_observable(self, term, coeff=1., k=10):
        """Returns the estimated observable for a term and its coefficient.

        Args:
            term (tuple): Openfermion style of a qubit operator term.
            coeff (float): Multiplication factor for the term.
            k (int): Grouping k observations for the means of median protocol.

        Returns:
            float: Observable estimated with the shadow.
        """

        shadow_size = len(self)
        dict_term = dict(term)
        observables_to_median = list()

        shadow_step = shadow_size // k

        # Median of average loop.
        for i_snapshot in range(0, shadow_size - shadow_step, shadow_step):
            observables_to_mean = np.empty(shadow_step, dtype=float)

            for j_snapshot in range(i_snapshot, i_snapshot + shadow_step):
                # Uses the fact that the trace of a tensor product is the
                # product of the traces of the indidual matrices
                # Tr(L \otimes M) = Tr(L) * Tr(M)
                # Also uses the fact that the product of two matrices that
                # are tensor products of smaller matrices is the tensor
                # products of the smaller matrices multiplied
                # (L \otimes M) @ (O \otimes P) = (L @ O) \otimes (M @ P)
                trace = 1.
                for i_qubit in range(self.n_qubits):
                    # If there is an operation applied on the qubit n.
                    if i_qubit in dict_term.keys():
                        obs = matrices[dict_term[i_qubit]]
                        tobs = traces[dict_term[i_qubit]]
                    else:
                        # Trace of identity matrix
                        obs = I
                        tobs = 2

                    state = zero_state if self.bitstrings[j_snapshot][i_qubit] == "0" else one_state
                    U = rotations[self.unitaries[j_snapshot][i_qubit]]

                    # Make <b_n|U_n for one qubit.
                    right_side = state @ U

                    # Make obs_n @ U_n^+ @ |b_n> for one qubit.
                    left_side = obs @ right_side.conj()

                    # Below is the faster way to compute
                    # trace *= np.trace(3*np.outer(left_side, right_side) - obs)
                    trace *= 3*np.dot(left_side, right_side) - tobs

                observables_to_mean[j_snapshot - i_snapshot] = np.real(coeff*trace)
            observables_to_median.append(np.mean(observables_to_mean))

        return np.median(observables_to_median)
