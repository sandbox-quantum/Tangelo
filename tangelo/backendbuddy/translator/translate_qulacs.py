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
    GATE_QULACS["S"] = qulacs.QuantumCircuit.add_S_gate
    GATE_QULACS["T"] = qulacs.QuantumCircuit.add_T_gate
    GATE_QULACS["RX"] = qulacs.QuantumCircuit.add_RX_gate
    GATE_QULACS["RY"] = qulacs.QuantumCircuit.add_RY_gate
    GATE_QULACS["RZ"] = qulacs.QuantumCircuit.add_RZ_gate
    GATE_QULACS["CNOT"] = qulacs.QuantumCircuit.add_CNOT_gate
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
            (GATE_QULACS[gate.name])(target_circuit, gate.target)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target, -1. * gate.parameter)
        elif gate.name in {"CNOT"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.control, gate.target)
        elif gate.name in {"MEASURE"}:
            gate = (GATE_QULACS[gate.name])(gate.target, gate.target)
            target_circuit.add_gate(gate)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qulacs")

        # Add noisy gates
        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    target_circuit.add_gate(Probabilistic(np, [X(gate.target), Y(gate.target), Z(gate.target)]))
                    if gate.control or gate.control == 0:
                        target_circuit.add_gate(Probabilistic(np, [X(gate.control), Y(gate.control), Z(gate.control)]))
                elif nt == 'depol':
                    if gate.control or gate.control == 0:
                        target_circuit.add_gate(TwoQubitDepolarizingNoise(gate.control, gate.target, (15/16)*np))
                    else:
                        target_circuit.add_gate(DepolarizingNoise(gate.target, (3/4) * np))

    return target_circuit
