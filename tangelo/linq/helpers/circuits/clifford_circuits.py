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

from math import pi, isclose

from tangelo.linq import Gate


def decompose_gate_to_cliffords(gate, abs_tol=1e-4):
    """
    Decomposes a single qubit parameterized gate into Clifford gates.

    Args:
        gate (Gate): The gate to be decomposed.
        abs_tol (float): Optional, absolute tolerance for value comparison (default: 1e-4).

    Returns:
        list: A list of Clifford gates representing the decomposition.

    Raises:
        ValueError: If parameterized gate cannot be decomposed into Clifford gates.

    """
    # Return an error if the gate is not in the Clifford gate set
    if not gate.is_clifford(abs_tol):
        raise ValueError(f"Error. The following gate cannot be decomposed into Clifford gates:\n {gate}")
    # If gate is not in the parameterized gate set, it decomposes to itself
    elif gate.name not in {"RX", "RY", "RZ", "PHASE"}:
        return gate
    # If gate parameter is close to 0, the gate operation is the identity
    elif isclose(gate.parameter, 0, abs_tol=abs_tol):
        return []

    # Find which Clifford parameter gate parameter corresponds to.
    clifford_values = [0, pi, pi / 2, -pi / 2]
    clifford_parameter = next((value for value in clifford_values if
                               isclose(gate.parameter % (2 * pi), value % (2 * pi), abs_tol=abs_tol)), None)

    if clifford_parameter is None:
        raise ValueError(f"Error: Parameterized gate {gate} cannot be decomposed into Clifford gates")

    gate_list = []
    if gate.name == "RY":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("H", gate.target), Gate("Z", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("Z", gate.target), Gate("H", gate.target)]

        elif clifford_parameter == pi:
            gate_list = [Gate("SDAG", gate.target), Gate("Y", gate.target), Gate("SDAG", gate.target)]

    elif gate.name == "RX":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("SDAG", gate.target), Gate("H", gate.target), Gate("SDAG", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("SDAG", gate.target), Gate("X", gate.target), Gate("SDAG", gate.target)]

    elif gate.name == "RZ":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("H", gate.target), Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target),
                         Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("SDAG", gate.target), Gate("H", gate.target),
                         Gate("SDAG", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("H", gate.target), Gate("SDAG", gate.target), Gate("X", gate.target), Gate("SDAG", gate.target), Gate("H", gate.target)]

    elif gate.name == "PHASE":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("SDAG", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("S", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Z", gate.target)]

    return gate_list
