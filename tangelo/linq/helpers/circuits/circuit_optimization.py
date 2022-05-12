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

"""Helper function: simple circuit optimization."""

from tangelo.linq import  Circuit


def remove_redundant_gates(circuit):
    """Docstring"""

    # Initial set of gates is the original one.
    gates = circuit._gates

    # Perform gate cancellation until no more is detected.
    while True:
        gate_indices_to_remove = list()
        last_gates = dict()

        # Loop through the updated list of gates.
        for gi, gate in enumerate(gates):

            # On which qubits this gate is acting on?
            qubits = gate.target if gate.control is None else gate.target + gate.control

            # Looping through the relevant qubits for this gate. If the last
            # gate acting on those qubits is the inverse (same target, control),
            # the gates can be removed. Otherwise, we store this gate as the
            # new last gate.
            to_remove = True
            for qubit_i in qubits:
                previous_gate_i, previous_gate = last_gates.get(qubit_i, (None, None))

                if previous_gate is None or previous_gate.inverse() != gate:
                    last_gates[qubit_i] = (gi, gate)
                    to_remove = False

            if to_remove:
                gate_indices_to_remove.extend([previous_gate_i, gi])

        # If no redundant gates are detected, break the loop.
        if len(gate_indices_to_remove) == 0:
            break

        # Remove the redundant gates for this pass.
        gates = [gate for gate_i, gate in enumerate(gates) if gate_i not in gate_indices_to_remove]

    return Circuit(gates)


def remove_small_rotations(circuit, param_threshold=0.05):
    """Docstring"""

    gate_indices_to_remove = list()

    # Looping through the gate. Only one pass is needed.
    for gi, gate in enumerate(circuit._gates):

        # If it is a rotation gate, and the angle is below the param_threshold,
        # the gate is removed. Further optimization must be done to remove the
        # CNOT ladders and basis rotation before/after the rotation.
        if gate.name in {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"} and abs(gate.parameter) < param_threshold:
            gate_indices_to_remove.append(gi)

    # Removal of the small rotation gates.
    gates = [gate for gate_i, gate in enumerate(circuit._gates) if gate_i not in gate_indices_to_remove]

    return Circuit(gates)
