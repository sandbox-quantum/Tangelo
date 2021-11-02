# Copyright 2021 1QB Information Technologies Inc.
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

"""This abstract quantum circuit class allows to represent a quantum circuit,
described by successive abstract quantum gate operations, without tying it to a
particular backend. It also provides methods to compute some of its
characteristics (width, size ...).
"""

from typing import List
from qsdk.backendbuddy import Gate


class Circuit:
    """An abstract quantum circuit class, represented by a list of abstract gate
    operations acting on qubits indices. From this list of gates, the gate
    counts, width and other properties can be computed. In the future, this
    information allows for finding unentangled registers or qubits only left in
    a classical state.

    The attributes of the Circuit class should not be modified by the user: the
    add_gate method will ensure their value is correct as gates are added to the
    circuit object.

    It is however ok to modify the value of the parameters for variational gate
    operations in a sub-class of Circuit, as this does not impact the quantum
    circuit information. An example of how to do this to engineer a
    parameterized ansatz to be used in a variational algorithm in available in
    the example folder.
    """

    def __init__(self, gates: List[Gate] = None, n_qubits=None):
        """Initialize gate list and internal variables depending on user input."""

        self._gates = list()
        self._qubits_simulated = n_qubits
        self._qubit_indices = set() if not n_qubits else set(range(n_qubits))
        self._gate_counts = dict()
        self._variational_gates = []

        if gates:
            _ = [self.add_gate(g) for g in gates]

    def __str__(self):
        """Print info about the circuit and the gates it contains in a
        human-friendly format.
        """

        mystr = f"Circuit object. Size {self.size} \n\n"
        for abs_gate in self._gates:
            mystr += abs_gate.__str__() + "\n"
        return mystr

    def __add__(self, other):
        """Concatenate the list of instructions of two circuit objects into a
        single one.
        """
        return Circuit(self._gates + other._gates, n_qubits=max(self.width, other.width))

    @property
    def size(self):
        """The size is the number of gates in the circuit. It is different from
        the depth.
        """
        return len(self._gates)

    @property
    def width(self):
        """Return the number of qubits required by this circuit. Assume all
        qubits are needed, even those who do not appear in a gate instruction.
        Qubit indexing starts at 0.
        """
        return max(self._qubit_indices) + 1 if self._qubit_indices else 0

    @property
    def counts(self):
        """Return the counts for all gate types included in the circuit."""
        return self._gate_counts

    @property
    def is_variational(self):
        """Returns true if the circuit holds any variational gate."""
        return True if self._variational_gates else False

    @property
    def is_mixed_state(self):
        """Assume circuit leads to a mixed state due to mid-circuit measurement
        if any MEASURE gate was explicitly added by the user.
        """
        return "MEASURE" in self.counts

    def add_gate(self, gate):
        """Add a new gate to a circuit object and update other fields of the
        circuit object to gradually keep track of its properties (gate count,
        qubit indices...).
        """
        # Add the gate to the list of gates
        self._gates += [gate]

        # A circuit is variational as soon as a variational gate is added to it
        if gate.is_variational:
            self._variational_gates.append(gate)

        def check_index_valid(index):
            """If circuit size was specified at instantiation, check that qubit
            indices do not go beyond it.
            """
            if self._qubits_simulated:
                if index >= self._qubits_simulated:
                    raise ValueError(f"Qubit index beyond expected maximal index ({self._qubits_simulated-1})\n "
                                     f"Gate = {gate}")

        # Track qubit indices
        self._qubit_indices.add(gate.target)
        check_index_valid(gate.target)
        if isinstance(gate.control, int):
            self._qubit_indices.add(gate.control)
            check_index_valid(gate.control)

        # Keep track of the total gate count
        self._gate_counts[gate.name] = self._gate_counts.get(gate.name, 0) + 1

    def serialize(self):
        return {"type": "QuantumCircuit", "gates": [gate.serialize() for gate in self._gates]}
