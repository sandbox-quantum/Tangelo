from math import pi, isclose

from tangelo.linq import Gate


def decompose_gate_to_cliffords(gate, abs_tol=1e-4):
    """
    Decomposes a single qubit parameterized gate into Clifford gates.

    Parameters:
    gate (Gate): The gate to be decomposed.
    abs_tol (float): Optional, absolute tolerance for value comparison (default: 1e-4).

    Returns:
    list: A list of Clifford gates representing the decomposition.

    Raises:
    ValueError: If parameterized gate cannot be decomposed into Clifford gates.

    """

    if gate.name not in {"RX", "RY", "RZ", "PHASE"}:
        return gate
    gate_list = []
    clifford_values = [0, pi, pi / 2, -pi / 2]
    value_isclose = [isclose(gate.parameter, value, abs_tol=abs_tol) for value in clifford_values]

    if not any(value_isclose):
        raise ValueError(
            f"Error: Parameterized gate {gate} cannot be decomposed into Clifford gates")
    else:
        clifford_parameter = [value for bool_, value in zip(value_isclose, clifford_values) if bool_][0]

    if clifford_parameter == 0:
        gate_list = [Gate("I", gate.target)]

    elif gate.name == "RY":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("Z", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("Z", gate.target)]
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