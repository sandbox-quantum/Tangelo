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

"""Functions helping with quantum circuit format conversion between abstract
format and qulacs format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""
from numpy import exp, cos, sin


def get_qulacs_gates():
    """Map gate name of the abstract format to the equivalent add_gate method of
    Qulacs's QuantumCircuit class API and supported gates:
    http://qulacs.org/class_quantum_circuit.html
    """
    import qulacs

    GATE_QULACS = dict()
    GATE_QULACS["H"] = qulacs.QuantumCircuit.add_H_gate
    GATE_QULACS["X"] = qulacs.QuantumCircuit.add_X_gate
    GATE_QULACS["Y"] = qulacs.QuantumCircuit.add_Y_gate
    GATE_QULACS["Z"] = qulacs.QuantumCircuit.add_Z_gate
    GATE_QULACS["CH"] = qulacs.gate.H
    GATE_QULACS["CX"] = qulacs.gate.X
    GATE_QULACS["CY"] = qulacs.gate.Y
    GATE_QULACS["CZ"] = qulacs.gate.Z
    GATE_QULACS["S"] = qulacs.QuantumCircuit.add_S_gate
    GATE_QULACS["T"] = qulacs.QuantumCircuit.add_T_gate
    GATE_QULACS["RX"] = qulacs.QuantumCircuit.add_RX_gate
    GATE_QULACS["RY"] = qulacs.QuantumCircuit.add_RY_gate
    GATE_QULACS["RZ"] = qulacs.QuantumCircuit.add_RZ_gate
    GATE_QULACS["CNOT"] = qulacs.QuantumCircuit.add_CNOT_gate
    GATE_QULACS["CRX"] = qulacs.gate.RX
    GATE_QULACS["CRY"] = qulacs.gate.RY
    GATE_QULACS["CRZ"] = qulacs.gate.RZ
    GATE_QULACS["PHASE"] = qulacs.gate.DenseMatrix
    GATE_QULACS["CPHASE"] = qulacs.gate.DenseMatrix
    GATE_QULACS["XX"] = qulacs.gate.DenseMatrix
    GATE_QULACS["SWAP"] = qulacs.QuantumCircuit.add_SWAP_gate
    GATE_QULACS["CSWAP"] = qulacs.gate.SWAP
    GATE_QULACS["MEASURE"] = qulacs.gate.Measurement
    return GATE_QULACS


def translate_qulacs(source_circuit, noise_model=None):
    """Take in an abstract circuit, return an equivalent qulacs QuantumCircuit
    instance. If provided with a noise model, will add noisy gates at
    translation. Not very useful to look at, as qulacs does not provide much
    information about the noisy gates added when printing the "noisy circuit".

    Args:
        source_circuit: quantum circuit in the abstract format.
        noise_model: A NoiseModel object from this package, located in the
            noisy_simulation subpackage.

    Returns:
        qulacs.QuantumCircuit: the corresponding qulacs quantum circuit.
    """

    import qulacs
    from qulacs.gate import X, Y, Z, Probabilistic, DepolarizingNoise, TwoQubitDepolarizingNoise

    GATE_QULACS = get_qulacs_gates()
    target_circuit = qulacs.QuantumCircuit(source_circuit.width)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target[0])
        elif gate.name in {"CH", "CX", "CY", "CZ"}:
            mat_gate = qulacs.gate.to_matrix_gate(GATE_QULACS[gate.name](gate.target[0]))
            for c in gate.control:
                mat_gate.add_control_qubit(c, 1)
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target[0], -1. * gate.parameter)
        elif gate.name in {"CRX", "CRY", "CRZ"}:
            mat_gate = qulacs.gate.to_matrix_gate(GATE_QULACS[gate.name](gate.target[0], -1. * gate.parameter))
            for c in gate.control:
                mat_gate.add_control_qubit(c, 1)
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"SWAP"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target[0], gate.target[1])
        elif gate.name in {"CSWAP"}:
            mat_gate = qulacs.gate.to_matrix_gate(GATE_QULACS[gate.name](gate.target[0], gate.target[1]))
            for c in gate.control:
                mat_gate.add_control_qubit(c, 1)
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"PHASE"}:
            mat_gate = GATE_QULACS[gate.name](gate.target[0], [[1, 0], [0, exp(1j * gate.parameter)]])
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"CPHASE"}:
            mat_gate = GATE_QULACS[gate.name](gate.target[0], [[1, 0], [0, exp(1j * gate.parameter)]])
            for c in gate.control:
                mat_gate.add_control_qubit(c, 1)
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"XX"}:
            c = cos(gate.parameter/2)
            s = -1j * sin(gate.parameter/2)
            mat_gate = GATE_QULACS[gate.name]([gate.target[0], gate.target[1]], [[c, 0, 0, s],
                                                                                 [0, c, s, 0],
                                                                                 [0, s, c, 0],
                                                                                 [s, 0, 0, c]])
            target_circuit.add_gate(mat_gate)
        elif gate.name in {"CNOT"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.control[0], gate.target[0])
        elif gate.name in {"MEASURE"}:
            gate = (GATE_QULACS[gate.name])(gate.target[0], gate.target[0])
            target_circuit.add_gate(gate)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qulacs")

        # Add noisy gates
        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    for t in gate.target:
                        target_circuit.add_gate(Probabilistic(np, [X(t), Y(t), Z(t)]))
                    if gate.control is not None:
                        for c in gate.control:
                            target_circuit.add_gate(Probabilistic(np, [X(c), Y(c), Z(c)]))
                elif nt == 'depol':
                    depol_list = [t for t in gate.target]
                    if gate.control is not None:
                        depol_list += [c for c in gate.control]
                    n_depol = len(depol_list)
                    if n_depol == 2:
                        target_circuit.add_gate(TwoQubitDepolarizingNoise(*depol_list, (15/16)*np))
                    elif n_depol == 1:
                        target_circuit.add_gate(DepolarizingNoise(depol_list[0], (3/4) * np))
                    else:
                        raise ValueError(f'{gate.name} has more than 2 qubits, Qulacs DepolarizingNoise only supports 1- and 2-qubits')

    return target_circuit
