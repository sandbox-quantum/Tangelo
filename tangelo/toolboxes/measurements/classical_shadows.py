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

"""This file provides functions to contruct and perform operation with classical
shadow.
"""

import abc
import numpy as np
import random
import warnings

from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of
from tangelo.toolboxes.operators.operators import QubitOperator

# TODO method or property to get the circuits

# Definition of important matrices used in observable or state estimation.

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

# Reversed channel to undo single Pauli rotations.
S_T = np.array([[1, 0], [0, -1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I = np.array([[1, 0], [0, 1]])
rotations = {"X": H, "Y": H @ S_T, "Z": I}


class ClassicalShadow(abc.ABC):
    """Abstract class for the classical shadow implementation. Classical shadow
    is a mean to characterize a quantum state (within an error treshold) with
    the fewest measurement possible. The original idea is described in
    H.Y. Huang, R. Kueng, and J. Preskill, Nature Physics 16, 1050 (2020).
    """

    def __init__(self, circuit, bitstrings=None, unitaries=None):
        """Default constructor for the classical shadow object. This class is
        the parent class for the different classical shadow flavors. The
        classical shadow object is defined by the bistrings and unitaries used
        in the process. Abstract methods are defined to take into account the
        procedure to inverse the channel.

        Args:
            bistrings (list of str): Representation of the outcomes for all
                snapshots. E.g. ["11011", "10000", ...].
            unitaries (list of str): REpresentation of the unitary for every
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
        return self.circuit.size

    @property
    def size(self):
        """Number of shots used to make the shadow."""
        return len(self.bitstrings)

    def __len__(self):
        """Same as the shadow size."""
        return self.size

    def append(self, bitstring, unitary):
        """Append method to merge a new snapshot to an existing shadow.

        Args:
            bistring (str): Representation of a single outcome.
            unitary (str): Relevant unitary for this single outcome.
        """
        if isinstance(bitstring, list) or isinstance(unitary, list):
            raise ValueError("Please use the extend method if bistring and unitary are lists.")
        self.bitstrings.append(bitstring)
        self.unitaries.append(unitary)

    def extend(self, bitstrings, unitaries):
        """Extend method to merge new snapshots to an existing shadow.

        Args:
            bistrings (list of str): Representation of a collection of outcomes.
            unitarys (list of str): Relevant unitaries for those outcomes.
        """
        if isinstance(bitstrings, str) or isinstance(unitaries, str):
            raise ValueError("Please use the append method if bistring and unitary are strings.")
        self.bitstrings += bitstrings
        self.unitaries += unitaries

    def get_circuits(self):
        """Docstring."""

        if not self.unitaries:
            raise ValueError(f"The build method of {self.__class__.__name__} must be called before simulation.")

        appended_circuits = list()

        for pauli_word in self.unitaries:
            # Transformation of a unitary to quantum gates.
            pauli_of = pauli_string_to_of(pauli_word)
            basis_circuit = Circuit(measurement_basis_gates(pauli_of), self.n_qubits)

            appended_circuits += [basis_circuit]

        return self.circuit, appended_circuits

    def get_observable(self, qubit_op, *args, **kwargs):
        """Getting an estimated observable value for a qubit operator from the
        classical shadow. This function loops through all terms and calls, for
        each one, the get_term_observable defined in the child class. Other
        arguments (args, kwargs) can be passed to the method.

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
            circuit (Circuit): State to characterize with a shadow.
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend.
        """

        if not self.unitaries:
            raise ValueError(f"The build method of {self.__class__.__name__} must be called before simulation.")

        if backend.n_shots != 1:
            warnings.warn(f"Warning: changing the number of shots to 1 for the backend.")
            backend.n_shots = 1

        for pauli_word in self.unitaries:
            # Transformation of a unitary to quantum gates.
            pauli_of = pauli_string_to_of(pauli_word)
            basis_circuit = Circuit(measurement_basis_gates(pauli_of), self.n_qubits)

            # Different behavior if circuit or initial_statevector is defined.
            if self.circuit is None and initial_statevector is not None:
                one_shot_circuit = basis_circuit if (basis_circuit.size > 0) else Circuit(n_qubits=self.n_qubits)
            elif self.circuit is not None and initial_statevector is None:
                one_shot_circuit = self.circuit + basis_circuit if (basis_circuit.size > 0) else self.circuit
            else:
                raise ValueError("A linq.Circuit or an initial_statevector must be provided.")

            results = backend.simulate(one_shot_circuit, initial_statevector=initial_statevector)

            # Output is of simulate is in the form ({'0100...': 1.0}, None).
            # The wanted bitstring is the only key in the first element of the
            # tuple.
            bitstring = list(results[0].keys())[0]

            # Appending the results to the shadow.
            self.bitstrings += [bitstring]

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def estimate_state(self):
        pass

    @abc.abstractmethod
    def get_term_observable(self):
        pass


class RandomizedClassicalShadow(ClassicalShadow):
    r"""Classical shadow using randomized single Pauli measurements, as defined
    in H.Y. Huang, R. Kueng, and J. Preskill, Nature Physics 16, 1050 (2020). In
    short, the channel is inversed to geet the state with the formula
    \hat{\rho} = \bigotimes_{j=1}^n \left( 3U_j^{\dagger} |b_j\rangle \langle b_j| U_j - \mathbb{I} \right)
    """

    def build(self, n_shots):
        """Random sampling of single pauli words.

        Args:
            n_shots (int): Total number of measurement.

        Returns:
            list of str: Measurements generated for a randomized procedure.
        """
        measurement_procedure = []
        for _ in range(n_shots):
            single_round_measurement = "".join([random.choice(["X", "Y", "Z"]) for _ in range(self.n_qubits)])
            measurement_procedure.append(single_round_measurement)

        self.unitaries = measurement_procedure
        return measurement_procedure

    def estimate_state(self, start=0, end=-1, list_of_index=None):
        """Returns classical shadow average density matrix for a range of
        snapshots.

        Args:
            start (int): Starting snapshot for the wanted range.
            end (int): Ending snapshot for the wanted range.
            list_of_index (list int): Specific snapshot to pick. If this
                variable is set, start and end are ignored.
        Returns:
            array of complex: Estimation of the 2^n * 2^n state.
        """

        # Ability to pick specific or a range of snapshots. Default behavior is
        # averaging over all snapshot.
        if list_of_index is not None:
            snapshot_index = list_of_index
        else:
            if end == -1:
                end = self.size
            snapshot_index = list(range(start, min(end, self.size)))

        # Creation of the density matrix object where snapshots rho will be
        # added to.
        rho = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)

        # Undoing rotations for the selected snapshot(s).
        for snapshot_i in snapshot_index:

            # Starting point is the Identity matrix of size 1.
            rho_snapshot = np.ones((1, 1), dtype=complex)

            for n in range(self.n_qubits):
                state = zero_state if self.bitstrings[snapshot_i][n] == "0" else one_state

                # Unitary to undo the rotation.
                U = rotations[self.unitaries[snapshot_i][n]]
                bU = state @ U
                rho_i = 3 * np.outer(bU.conj(), bU) - I
                rho_snapshot = np.kron(rho_snapshot, rho_i)

            rho += rho_snapshot[:, :]

        # Saving the result.
        self.state_estimate = rho / len(snapshot_index)

        # Returning the state estimation.
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

        # Median of average loop.
        for i in range(0, shadow_size, shadow_size // k):
            observables_to_mean = np.empty(shadow_size//k, dtype=float)

            if i + shadow_size//k <= shadow_size:
                for snapshot in range(i, i + shadow_size // k):
                    # Uses the fact that the trace of a tensor product is the
                    # product of the traces of the indidual matrices
                    # Tr(L \otimes M) = Tr(L) * Tr(M)
                    # Also uses the fact that the product of two matrices that
                    # are tensor products of smaller matrices is the tensor
                    # products of the smaller matrices multiplied
                    # (L \otimes M) @ (O \otimes P) = (L @ O) \otimes (M @ P)
                    trace = 1.
                    for n in range(self.n_qubits):
                        # If there is an operation applied on the qubit n.
                        if n in dict_term.keys():
                            obs = matrices[dict_term[n]]
                            tobs = traces[dict_term[n]]

                            state = zero_state if self.bitstrings[snapshot][n] == "0" else one_state
                            U = rotations[self.unitaries[snapshot][n]]

                            # Make <b_n|U_n for one qubit.
                            right_side = np.matmul(state, U)

                            # Make obs_n @ U_n^+ @ |b_n> for one qubit.
                            left_side = right_side.conj()
                            left_side = np.matmul(obs, left_side)

                            # Below is the faster way of trace *= np.trace(
                            # 3*np.outer(left_side, right_side) - obs).
                            trace *= 3*np.dot(left_side, right_side) - tobs
                        # If there is no operation applied on the qubit n.
                        else:
                            # Trace of identity matrix
                            obs = I
                            tobs = 2

                    observables_to_mean[snapshot - i] = np.real(coeff*trace)
                observables_to_median.append(np.mean(observables_to_mean))

        return np.median(observables_to_median)
