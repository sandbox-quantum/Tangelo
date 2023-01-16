from math import pi, isclose
import numpy as np
from random import sample

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.ansatz_generator.variational_circuit import VariationalCircuitAnsatz

def get_clifford_circuit(circuit, args = "nearest"):
    """Returns Clifford version of circuit by setting all RX, RY, RZ and PHASE to corresponding clifford values
    Args:
        circuit (Circuit):
        args: "nearest", "random", or [list]

    Returns:
        Circuit: Clifford version of circuit
    """
    ansatz = VariationalCircuitAnsatz(circuit)
    clifford_values = [0, pi, pi / 2, -pi / 2, -pi]

    if args == "nearest":
        var_params = [np.arctan2(np.sin(angle), np.cos(angle)) for angle in ansatz.var_params_default]
        nearest_params = [min(clifford_values, key=lambda x: abs(x - value)) for value in var_params]
        ansatz.update_var_params(nearest_params)

    if args == "random":
        ansatz.update_var_params(sample(clifford_values, ansatz.n_var_params))

    if isinstance(args, list):
        ansatz.update_var_params(args)

    return ansatz.circuit

def decompose_gate_to_cliffords(gate, abs_tol=1e-4):
    """For specific parameters, this function will decompose a single qubit parameterized gate into clifford gates"""

    gate_list = []

    clifford_values = [0, pi, pi / 2, -pi / 2]
    value_isclose = [isclose(gate.parameter, value, abs_tol=abs_tol) for value in clifford_values]

    if not any(value_isclose):
        raise ValueError(
            f"Error: Gate {gate} cannot be decomposed into Clifford gates")
    else:
        clifford_parameter = [value for bool_, value in zip(value_isclose, clifford_values) if bool_]

    if clifford_parameter == 0:
        gate_list = [Gate("I", gate.target)]

    elif gate.name == "RY":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("Z", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("Z", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Y", gate.target)]

    elif gate.name == "RX":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("SDAG", gate.target), Gate("H", gate.target), Gate("SDAG", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("X", gate.target)]

    elif gate.name == "RZ":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("H", gate.target), Gate("S", gate.target), Gate("H", gate.target), Gate("S", gate.target),
                         Gate("H", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("H", gate.target), Gate("SDAG", gate.target), Gate("H", gate.target),
                         Gate("SDAG", gate.target), Gate("H", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Z", gate.target)]

    elif gate.name == "PHASE":
        if clifford_parameter == -pi / 2:
            gate_list = [Gate("SDAG", gate.target)]
        elif clifford_parameter == pi / 2:
            gate_list = [Gate("S", gate.target)]
        elif clifford_parameter == pi:
            gate_list = [Gate("Z", gate.target)]

    return gate_list