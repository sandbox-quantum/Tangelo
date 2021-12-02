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
format and Braket format

In order to produce an equivalent circuit for the target backend, it is necessary
to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""


def get_braket_gates():
    """Map gate name of the abstract format to the equivalent methods of the
    braket.circuits.Circuit class API and supported gates:
    https://amazon-braket-sdk-python.readthedocs.io/en/latest/_apidoc/braket.circuits.circuit.html
    """

    from braket.circuits import Circuit as BraketCircuit

    GATE_BRAKET = dict()
    GATE_BRAKET["H"] = BraketCircuit.h
    GATE_BRAKET["X"] = BraketCircuit.x
    GATE_BRAKET["Y"] = BraketCircuit.y
    GATE_BRAKET["Z"] = BraketCircuit.z
    GATE_BRAKET["CX"] = BraketCircuit.cnot
    GATE_BRAKET["CY"] = BraketCircuit.cy
    GATE_BRAKET["CZ"] = BraketCircuit.cz
    GATE_BRAKET["S"] = BraketCircuit.s
    GATE_BRAKET["T"] = BraketCircuit.t
    GATE_BRAKET["RX"] = BraketCircuit.rx
    GATE_BRAKET["RY"] = BraketCircuit.ry
    GATE_BRAKET["RZ"] = BraketCircuit.rz
    GATE_BRAKET["XX"] = BraketCircuit.xx
    GATE_BRAKET["CRZ"] = [BraketCircuit.cphaseshift, BraketCircuit.cphaseshift10]
    GATE_BRAKET["PHASE"] = BraketCircuit.phaseshift
    GATE_BRAKET["CPHASE"] = BraketCircuit.cphaseshift
    GATE_BRAKET["CNOT"] = BraketCircuit.cnot
    GATE_BRAKET["SWAP"] = BraketCircuit.swap
    GATE_BRAKET["CSWAP"] = BraketCircuit.cswap
    # GATE_BRAKET["MEASURE"] = ? (mid-circuit measurement currently unsupported?)

    return GATE_BRAKET


def translate_braket(source_circuit):
    """Take in an abstract circuit, return a quantum circuit object as defined
    in the Python Braket SDK.

    Args:
        source_circuit: quantum circuit in the abstract format.

    Returns:
        braket.circuits.Circuit: quantum circuit in Python Braket SDK format.
    """

    from braket.circuits import Circuit as BraketCircuit

    GATE_BRAKET = get_braket_gates()
    target_circuit = BraketCircuit()

    # Map the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.control is not None:
            if len(gate.control) > 1:
                raise ValueError('Multi-controlled gates not supported with braket: Gate {gate.name} with controls {gate.control} is invalid')
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target[0])
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target[0], gate.parameter)
        elif gate.name in {"CNOT", "CX", "CY", "CZ"}:
            (GATE_BRAKET[gate.name])(target_circuit, control=gate.control[0], target=gate.target[0])
        elif gate.name in {"XX"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target[0], gate.target[1], gate.parameter)
        elif gate.name in {"CRZ"}:
            (GATE_BRAKET[gate.name][0])(target_circuit, gate.control[0], gate.target[0], gate.parameter/2.)
            (GATE_BRAKET[gate.name][1])(target_circuit, gate.control[0], gate.target[0], -gate.parameter/2.)
        elif gate.name in {"SWAP"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target[0], gate.target[1])
        elif gate.name in {"CSWAP"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.control[0], gate.target[0], gate.target[1])
        elif gate.name in {"CPHASE"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.control[0], gate.target[0], gate.parameter)
        # elif gate.name in {"MEASURE"}:
        # implement if mid-circuit measurement available through Braket later on
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend braket")
    return target_circuit
