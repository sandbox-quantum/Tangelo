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

"""This abstract quantum circuit class allows to represent a quantum circuit,
described by successive abstract quantum gate operations, without tying it to a
particular backend. It also provides methods to compute some of its
characteristics (width, size ...).
"""

import copy
from typing import List

import numpy as np

from tangelo.linq import Gate


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

    def __init__(self, gates: List[Gate] = None, n_qubits=None, name="no_name"):
        """Initialize gate list and internal variables depending on user input."""

        self.name = name
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

    def __mul__(self, n_repeat):
        """Return a circuit consisting of n_repeat repetitions of the input circuit.
        """
        if not isinstance(n_repeat, (int, np.integer)) or n_repeat <= 0:
            raise ValueError("Multiplication (repetition) operator with Circuit class only works for integers > 0")
        return Circuit(self._gates * n_repeat, n_qubits=self.width)

    def __rmul__(self, n_repeat):
        """Return a circuit consisting of n_repeat repetitions of the input circuit (circuit as right-hand side)
        """
        return self * n_repeat

    def __eq__(self, other):
        """Define equality (==) between 2 circuits. They are equal iff all their gates are equal, and they have
        the same numbers of qubits.
        """
        return (self._gates == other._gates) and (self.width == other.width)

    def __ne__(self, other):
        """Define inequality (!=) operator on circuits
        """
        return not (self == other)

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

    def add_gate(self, g):
        """Add a new gate to a circuit object and update other fields of the
        circuit object to gradually keep track of its properties (gate count,
        qubit indices...).
        """
        # Add a copy of the gate to the list of gates
        gate = Gate(g.name, g.target, g.control, g.parameter, g.is_variational)
        self._gates.append(gate)

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
        all_involved_qubits = gate.target if gate.control is None else gate.target + gate.control
        for q in all_involved_qubits:
            check_index_valid(q)
            self._qubit_indices.add(q)

        # Keep track of the total gate count
        self._gate_counts[gate.name] = self._gate_counts.get(gate.name, 0) + 1

    def depth(self):
        """ Return the depth of the quantum circuit, by computing the number of moments. Does not count
        qubit initialization as a moment (unlike Cirq, for example).
        """
        moments = list()

        for g in self._gates:
            qubits = set(g.target) if g.control is None else set(g.target + g.control)

            if not moments:
                moments.append(qubits)
            else:
                # Find latest moment involving at least one of the qubits targeted by the current gate
                # The current gate is part of the moment following that one
                for i, m in reversed(list(enumerate(moments))):
                    if m & qubits:
                        if (i+1) < len(moments):
                            moments[i+1] = moments[i+1] | qubits
                        else:
                            moments.append(qubits)
                        break
                    # Case where none of the target qubits have been used before
                    elif i == 0:
                        moments[0] = moments[0] | qubits

        return len(moments)

    def trim_qubits(self):
        """Trim unnecessary qubits and update indices with the lowest values possible.
        """
        qubits_in_use = set().union(*self.get_entangled_indices())
        mapping = {ind: i for i, ind in enumerate(qubits_in_use)}
        for g in self._gates:
            g.target = [mapping[ind] for ind in g.target]
            if g.control:
                g.control = [mapping[ind] for ind in g.control]

        self._qubit_indices = set(range(len(qubits_in_use)))
        return self

    def reindex_qubits(self, new_indices):
        """Reindex qubit indices according to users labels / new indices. The
        new indices are set according to [new_index_for_qubit0,
        nex_index_for_qubit1, ..., new_index_for_qubit_N] where new_index < N.
        """

        if len(new_indices) != len(self._qubit_indices):
            raise ValueError(f"The number of indices does not match the length of self._qubit_indices")

        qubits_in_use = self._qubit_indices
        mapping = {i: j for i, j in zip(qubits_in_use, new_indices)}
        for g in self._gates:
            g.target = [mapping[ind] for ind in g.target]
            if g.control:
                g.control = [mapping[ind] for ind in g.control]

        self._qubit_indices = set(new_indices)

    def get_entangled_indices(self):
        """Return a list of unentangled sets of qubit indices. Each set includes indices
         of qubits that form an entangled subset.
        """

        entangled_indices = list()
        for g in self._gates:
            # Gradually accumulate entangled indices from the different subsets
            # Remove and replace them with their union, for each gate.
            q_new = set(g.target) if g.control is None else set(g.target + g.control)
            for qs in entangled_indices[::-1]:
                if q_new & qs:
                    q_new = q_new | qs
                    entangled_indices.remove(qs)
            entangled_indices.append(q_new)

        return entangled_indices

    def split(self):
        """ Split a circuit featuring unentangled qubit subsets into as many circuit objects.
        Each circuit only contains the gate operations targeting the qubit indices in its subsets.

        Returns:
            list of Circuit: list of resulting circuits
        """
        entangled_indices = self.get_entangled_indices()
        separate_circuits = [Circuit() for i in range(len(entangled_indices))]
        for g in self._gates:
            q_new = set(g.target) if g.control is None else set(g.target + g.control)
            # Append the gate to the circuit that handles the corresponding qubit indices
            for i, indices in enumerate(entangled_indices):
                if q_new & indices:
                    separate_circuits[i].add_gate(g)
                    break

        # Trim unnecessary indices in the new circuits
        for c in separate_circuits:
            c.trim_qubits()
        return separate_circuits

    def stack(self, *other_circuits):
        """Convenience method to stack other circuits on top of this one.
        See separate stack function.

        Args:
            *other_circuits (Circuit): one or several circuit objects to stack

        Returns:
            Circuit: the stacked circuit
        """
        return stack(self, *other_circuits)

    def inverse(self):
        """Return the inverse (adjoint) of a circuit

        This is performed by reversing the Gate order and applying the inverse to each Gate.

        Returns:
            Circuit: the inverted circuit
        """
        gates = [gate.inverse() for gate in reversed(self._gates)]
        return Circuit(gates, n_qubits=self.width)

    def serialize(self):
        if not isinstance(self.name, str):
            return TypeError("Name of circuit object must be a string")
        return {"name": self.name, "type": "QuantumCircuit", "gates": [gate.serialize() for gate in self._gates]}

    def remove_small_rotations(self, param_threshold=0.05):
        """Convenience method to remove small rotations from the circuit.
        See separate remove_small_rotations function.

        Args:
            param_threshold (float): Max absolute value to consider a rotation
                as a small one.

        Returns:
            Circuit: The circuit without small rotations.
        """
        opt_circuit = remove_small_rotations(self, param_threshold)
        self.__dict__ = opt_circuit.__dict__

    def remove_redundant_gates(self):
        """Convenience method to remove redundant gates from the circuit.
        See separate remove_redundant_gates function.

        Returns:
            Circuit: The circuit without redundant gates.
        """
        opt_circuit = remove_redundant_gates(self)
        self.__dict__ = opt_circuit.__dict__


def stack(*circuits):
    """ Take list of circuits as input, and stack them (e.g concatenate them along the
    width (qubits)) to form a single wider circuit, which allows users to run all of
    these circuits at once on a quantum device.

    Stacking provides a way to "fill up" a device if many qubits would be unused otherwise,
    therefore reducing cost / duration of a hardware experiment. However, depending on the
    device, this may amplify some sources of errors or make qubit placement more challenging.

    Args:
        *circuits (Circuit): the circuits to trim and stack into a single one

    Returns:
        Circuit: the stacked circuit
    """

    if not circuits:
        return Circuit()

    # Trim qubits of input circuit for maximum compactness
    circuits = [c.trim_qubits() for c in copy.deepcopy(list(circuits))]

    # Stack circuits. Reindex each circuit with the proper offset and then concatenate, until done
    stacked_circuit = circuits.pop(0)
    for c in circuits:
        c_stack = copy.deepcopy(c)
        c_stack.reindex_qubits(list(range(stacked_circuit.width, stacked_circuit.width + c.width)))
        stacked_circuit += c_stack

    return stacked_circuit


def remove_small_rotations(circuit, param_threshold=0.05):
    """Remove small rotation gates, up to a parameter threshold, from the
    circuit. Rotations from the set {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"} are
    considered.

    Args:
        circuit (Circuit): the circuits to trim and stack into a single one
        param_threshold (float): Max absolute value to consider a rotation as
            a small one.

    Returns:
        Circuit: The circuit without small-rotation gates.
    """

    rot_gates = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"}
    gates = [g for g in circuit._gates if not (g.name in rot_gates and abs(g.parameter) % (2*np.pi) < param_threshold)]

    return Circuit(gates)


def remove_redundant_gates(circuit):
    """Remove redundant gates in a circuit. Redundant gates are adjacent gates
    that can be cancelled as their global effect is the identity. This function
    also works with many-qubit gates. However, it does not perform reordering of
    commutating gates to perform additional cancellations.

    Args:
        circuit (Circuit): the circuits to remove redundant gates.

    Returns:
        Circuit: The circuit without redundant gates.
    """
    gate_qubits = {i: list() for i in range(circuit.width)}
    indices_to_remove = list()

    for gi, gate in enumerate(circuit._gates):
        remove_gate = True

        # Identify qubits the current gate acts on.
        qubits = gate.target if gate.control is None else gate.target + gate.control

        # Check if the last gate cancels the current gate.
        for qubit_i in qubits:
            if not gate_qubits[qubit_i] or gate_qubits[qubit_i][-1][1].inverse() != gate:
                remove_gate = False
                break

        # Pop the last gate if the gate is to be removed.
        # If not, append the gate to the gate_qubits list.
        if remove_gate:
            indices_to_remove += [gi, gate_qubits[qubits[0]][-1][0]]
            for qubit_i in qubits:
                del gate_qubits[qubit_i][-1]
        else:
            for qubit_i in qubits:
                gate_qubits[qubit_i] += [(gi, gate)]

    # Remove gates that can be cancelled.
    gates = [gate for gate_i, gate in enumerate(circuit._gates) if gate_i not in indices_to_remove]

    return Circuit(gates)
