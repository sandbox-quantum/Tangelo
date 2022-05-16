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

"""Provide useful functions, corresponding to common patterns in quantum
chemistry circuits (CNOT ladders, Pauli-word to circuit translation ...) to
facilitate the assembly of ansatz quantum circuits.
"""

from copy import deepcopy
from itertools import combinations

import numpy as np
from openfermion.ops import FermionOperator as ofFermionOperator
from openfermion.ops import InteractionOperator as ofInteractionOperator
from openfermion.ops import QubitOperator as ofQubitOperator

from tangelo.linq import Circuit, Gate
from tangelo.toolboxes.operators import FermionOperator, QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping, get_fermion_operator


def pauli_op_to_gate(index, op, inverse=False):
    """Return the change-of-basis gates required to map pauli words to quantum
    circuit as per Whitfield 2010 (https://arxiv.org/pdf/1001.3855.pdf).
    """
    if op == "X":
        return Gate("H", index)
    elif op == "Y":
        gate = Gate("RX", index, parameter=0.5*np.pi)
        return gate if not inverse else gate.inverse()


def exp_pauliword_to_gates(pauli_word, coef, variational=True, control=None):
    """Generate a list of Gate objects corresponding to the exponential of a pauli word.
    The process is described in Whitfield 2010 https://arxiv.org/pdf/1001.3855.pdf

    Args:
        pauli_word (tuple): Openfermion-like tuple that generates a pauli_word to exponentiate
        coef (float): The coefficient in the exponentiation
        variational (bool): When creating the Gate objects, label the (controlled-)Rz gate as variational
        control (integer): The control qubit label

    Returns:
        list: list of Gate objects that represents the exponentiation of the pauli word.
    """
    gates = []

    # Before CNOT ladder
    for index, op in pauli_word:
        if op in {"X", "Y"}:
            gates += [pauli_op_to_gate(index, op, inverse=False)]

    # CNOT ladder and rotation
    indices = sorted([index for index, op in pauli_word])
    cnot_ladder_gates = [Gate("CNOT", target=pair[1], control=pair[0]) for pair in zip(indices[:-1], indices[1:])]
    gates += cnot_ladder_gates

    angle = 2.*coef if coef >= 0. else 4*np.pi+2*coef
    if control is None:
        gates += [Gate("RZ", target=indices[-1], parameter=angle, is_variational=variational)]
    else:
        gates += [Gate("CRZ", target=indices[-1], control=control, parameter=angle)]

    gates += cnot_ladder_gates[::-1]

    # After CNOT ladder
    for index, op in pauli_word[::-1]:
        if op in {"X", "Y"}:
            gates += [pauli_op_to_gate(index, op, inverse=True)]

    return gates


def get_exponentiated_qubit_operator_circuit(qubit_op, time=1., variational=False, trotter_order=1, control=None,
                                             return_phase=False, pauli_order=None):
    """Generate the exponentiation of a qubit operator in first- or second-order Trotterized form.
    The algorithm is described in Whitfield 2010 https://arxiv.org/pdf/1001.3855.pdf

    Args:
        qubit_op  (QubitOperator):  qubit hamiltonian to exponentiate
        time (float or dict): The time to evolve the whole system or individiual times for each
            term in the operator. If a dictionary, must have keys that have a matching key in qubit_op.terms
        variational (bool) : Whether the coefficients are variational
        trotter_order (int): order of trotter approximation, only 1 or 2 are supported.
        return_phase (bool): Return the global-phase generated
        pauli_order (list): The desired pauli_word order for trotterization defined as a list of (pauli_word, coeff)
            elements which have matching dictionary elements pauli_word: coeff in QubitOperator terms.items().
            The coeff in pauli_order is used to generate the exponential.

    Returns:
        Circuit: circuit corresponding to exponentiation of qubit operator
        phase : The global phase of the time evolution if return_phase=True else not included
    """
    if pauli_order is None:
        pauli_words = list(qubit_op.terms.items())
    elif isinstance(pauli_order, list):
        pauli_words = pauli_order.copy()
    else:
        raise ValueError("ordered terms must be a list with elements (keys, values) of qubit_op.terms.items()")

    if trotter_order > 2:
        raise ValueError(f"Trotter order of >2 is not supported currently in Tangelo.")
    prefactor = 1/2 if trotter_order == 2 else 1

    if isinstance(time, (float, np.floating, np.integer, int)):
        evolve_time = {term: prefactor*time for term in qubit_op.terms.keys()}
    elif isinstance(time, dict):
        if time.keys() == qubit_op.terms.keys():
            evolve_time = {term: prefactor*etime for term, etime in time.items()}
        else:
            raise ValueError(f"The keys in the time dictionary do not match the keys in qubit_op.terms")
    else:
        raise ValueError(f"time must be a float or a dictionary")

    phase = 1.
    exp_pauli_word_gates = list()
    for i in range(trotter_order):
        if i == 1:
            pauli_words.reverse()
        for pauli_word, coef in pauli_words:
            if pauli_word:  # identity terms do not contribute to evolution outside of a phase
                if abs(np.real(coef)*evolve_time[pauli_word]) > 1.e-10:
                    exp_pauli_word_gates += exp_pauliword_to_gates(pauli_word,
                                                                   np.real(coef)*evolve_time[pauli_word],
                                                                   variational=variational,
                                                                   control=control)
            else:
                if control is None:
                    phase *= np.exp(-1j * coef * evolve_time[pauli_word])
                else:
                    exp_pauli_word_gates += [Gate("PHASE", target=control, parameter=-np.real(coef)*evolve_time[pauli_word])]

    return_value = (Circuit(exp_pauli_word_gates), phase) if return_phase else Circuit(exp_pauli_word_gates)
    return return_value


def trotterize(operator, time=1., n_trotter_steps=1, trotter_order=1, variational=False,
               mapping_options=dict(), control=None, return_phase=False):
    """Generate the circuit that represents time evolution of an operator.
    This circuit is generated as a trotterization of a qubit operator which is either the input
    or mapped from the given fermion operator.

    Args:
        operator  (QubitOperator or FermionOperator):  operator to time evolve
        time (float or dict): The time to evolve the whole system or individiual times for each
            term in the operator. If a dict, each key must match the keys in operator.terms
        variational (bool): whether the coefficients are variational
        trotter_order (int): order of trotter approximation, 1 or 2 supported
        n_trotter_steps (int): The number of different time steps taken for total time t
        mapping_options (dict): Defines the desired Fermion->Qubit mapping
                                Default values:{"up_then_down": False, "qubit_mapping": "jw", "n_spinorbitals": None,
                                                "n_electrons": None}
        control (int): The label for the control Qubit of the time-evolution
        return_phase (bool): If return_phase is True, the global phase of the time-evolution will be returned

    Returns:
        Circuit: circuit corresponding to time evolution of the operator
        float: the global phase not included in the circuit if return_phase=True else not included
    """

    if isinstance(operator, ofFermionOperator):
        options = {"up_then_down": False, "qubit_mapping": "jw", "n_spinorbitals": None, "n_electrons": None}
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in mapping_options.items():
            if k in options:
                options[k] = v
            else:
                raise KeyError(f"Keyword :: {k}, not a valid fermion to qubit mapping option")
        if isinstance(time, (float, np.floating, int, np.integer)):
            evolve_time = {term: time for term in operator.terms.keys()}
        elif isinstance(time, dict):
            if time.keys() == operator.terms.keys():
                evolve_time = deepcopy(time)
            else:
                raise ValueError(f"keys of time do not match keys of operator.terms")
        else:
            raise ValueError("time must be a float or dictionary of floats")
        new_operator = FermionOperator()
        for term in operator.terms:
            new_operator += FermionOperator(term, operator.terms[term]*evolve_time[term]/n_trotter_steps)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=new_operator,
                                            mapping=options["qubit_mapping"],
                                            n_spinorbitals=options["n_spinorbitals"],
                                            n_electrons=options["n_electrons"],
                                            up_then_down=options["up_then_down"])
        circuit, phase = get_exponentiated_qubit_operator_circuit(qubit_op,
                                                                  time=1.,  # time is already included
                                                                  trotter_order=trotter_order,
                                                                  variational=variational,
                                                                  control=control,
                                                                  return_phase=True)

    elif isinstance(operator, (ofQubitOperator)):
        qubit_op = deepcopy(operator)
        if isinstance(time, float):
            evolve_time = time / n_trotter_steps
        elif isinstance(time, dict):
            if time.keys() == operator.terms.keys():
                evolve_time = {term: etime / n_trotter_steps for term, etime in time.items()}
            else:
                raise ValueError(f"time dictionary and operator.terms dictionary have different keys.")
        else:
            raise ValueError(f"time must be a float or a dictionary of floats")
        circuit, phase = get_exponentiated_qubit_operator_circuit(qubit_op,
                                                                  time=evolve_time,
                                                                  trotter_order=trotter_order,
                                                                  variational=variational,
                                                                  control=control,
                                                                  return_phase=True)
    else:
        raise ValueError("Only FermionOperator or QubitOperator allowed")

    return_value = (circuit*n_trotter_steps, phase**n_trotter_steps) if return_phase else circuit*n_trotter_steps
    return return_value


def append_qft_rotations_gates(gate_list, qubit_list, prefac=1):
    """Appends the list of gates required for a quantum fourier transform to a gate list.

    Args:
        gate_list (list): List of Gate elements
        qubit_list (list): List of integers for which the qft operations are performed

    Returns:
        list: List of Gate objects for rotation portion of qft circuit appended to gate_list"""
    n = len(qubit_list)
    if n == 0:
        return gate_list
    n -= 1
    gate_list += [Gate("H", target=qubit_list[n])]
    for i, qubit in enumerate(qubit_list[:n]):
        gate_list += [Gate("CPHASE", control=qubit, target=qubit_list[n], parameter=prefac*np.pi/2**(n-i))]

    append_qft_rotations_gates(gate_list, qubit_list[:n], prefac=prefac)


def swap_registers(gate_list, qubit_list):
    """Function to swap register order.
    Args:
        gate_list (list): List of Gate
        qubit_list (list): List of integers for the locations of the qubits

    Result:
        list: The Gate operations that swap the register order"""
    n = len(qubit_list)
    for qubit_index in range(n//2):
        gate_list += [Gate("SWAP", target=[qubit_list[qubit_index], qubit_list[n - qubit_index - 1]])]
    return gate_list


def get_qft_circuit(qubits, n_qubits=None, inverse=False, swap=True):
    """Returns the QFT or iQFT circuit given a list of qubits to act on.

    Args:
        qubits (int or list): The list of qubits to apply the QFT circuit to. If an integer,
            the operation is applied to the [0,...,qubits-1] qubits
        n_qubits: Argument to initialize a Circuit with the desired number of qubits.
        inverse (bool): If True, the inverse QFT is applied. If False, QFT is applied
        swap (bool): Whether to apply swap to the registers.

        Returns:
            Circuit: The circuit that applies QFT or iQFT to qubits
        """

    if isinstance(qubits, int):
        qubit_list = list(range(qubits))
    elif isinstance(qubits, list):
        qubit_list = qubits
    else:
        raise KeyError("qubits must be an int or list of ints")

    swap_gates = list()
    if swap:
        swap_registers(swap_gates, qubit_list)

    qft_gates = list()
    if inverse:
        append_qft_rotations_gates(qft_gates, qubit_list, prefac=-1)
        qft_gates = [gate for gate in reversed(qft_gates)]
        qft_gates = swap_gates + qft_gates
    else:
        append_qft_rotations_gates(qft_gates, qubit_list)
        qft_gates += swap_gates

    return Circuit(qft_gates, n_qubits=n_qubits)


def controlled_pauliwords(qubit_op, control, n_qubits=None):
    """Takes a qubit operator and returns controlled-pauliword circuits for each term as a list.

    Args:
        qubit_op (QubitOperator): The qubit operator with pauliwords to generate circuits for
        control (int): The index of the control qubit
        n_qubits (int): When generating each Circuit, create with n_qubits size

    Returns:
        list: List of controlled-pauliword Circuit for each pauliword in the qubit_op
    """
    pauli_words = qubit_op.terms.items()

    pauliword_circuits = list()
    for (pauli_word, _) in pauli_words:
        gates = [Gate(name="C"+op, target=index, control=control) for index, op in pauli_word]
        pauliword_circuits.append(Circuit(gates, n_qubits=n_qubits))
    return pauliword_circuits


def controlled_swap_to_XX_gates(c, n1, n2):
    """Equivalent decomposition of controlled swap into 1-qubit gates and XX 2-qubit gate.

    This is useful for IonQ experiments as the native two-qubit gate is the XX Ising coupling.

    Args:
        c (int): control qubit
        n1 (int): first target qubit
        n2 (int): second target qubit

    Returns:
        list: List of Gate that applies controlled swap operation
    """
    gates = [Gate("RY", target=c, parameter=7*np.pi/2.),
             Gate("RZ", target=n1, parameter=7*np.pi/2.),
             Gate("XX", target=[n1, n2], parameter=5*np.pi/2.),
             Gate("RZ", target=n1, parameter=7*np.pi/4.),
             Gate("RZ", target=n2, parameter=3*np.pi/4.),
             Gate("RY", target=n1, parameter=np.pi/2.),
             Gate("XX", target=[c, n2], parameter=7*np.pi/2.),
             Gate("RY", target=n2, parameter=11*np.pi/4),
             Gate("XX", target=[n1, n2], parameter=7*np.pi/2.),
             Gate("XX", target=[c, n1], parameter=np.pi/4.),
             Gate("RZ", target=n2, parameter=np.pi/4),
             Gate("XX", target=[c, n2], parameter=5*np.pi/2),
             Gate("RY", target=c, parameter=5*np.pi/2),
             Gate("RZ", target=n1, parameter=5*np.pi/2),
             Gate("RY", target=n2, parameter=7*np.pi/4),
             Gate("XX", target=[n1, n2], parameter=7*np.pi/2),
             Gate("RY", target=n1, parameter=np.pi/2),
             Gate("RZ", target=c, parameter=11*np.pi/4)]
    return gates


def derangement_circuit(qubit_list, control=None, n_qubits=None, decomp=None):
    """Returns the (controlled-)derangement circuit for multiple copies of a state

    Args:
        qubit_list (list of list(int)): Each item in the list is a list of qubit registers for each copy. The length of
            each list of qubit registers must be the same.
            For example [[1, 2], [3, 4]] applies controlled-swaps between equivalent states located on qubits [1, 2] and [3, 4]
        control (int): The control register to be measured.
        n_qubits (int): The number of qubits in the circuit.
        decomp (str): Use the decomposed controlled-swap into 1-qubit gates and a certain 2-qubit gate listed below.
            "XX": 2-qubit gate is XX

    Returns:
        Circuit: The derangement circuit
    """
    if decomp is not None and decomp not in ["XX"]:
        raise ValueError(f"{decomp} is not a valid controlled swap decomposition")

    num_copies = len(qubit_list)
    if num_copies == 1:
        return Circuit(n_qubits=n_qubits)
    else:
        rho_range = len(qubit_list[0])
        for i in range(1, num_copies):
            if len(qubit_list[i]) != rho_range:
                raise ValueError("All copies must have the same number of qubits")
    gate_list = list()
    if control is None:
        for copy1, copy2 in combinations(range(num_copies), 2):
            for rhoi in range(rho_range):
                gate_list += [Gate("SWAP", target=[qubit_list[copy1][rhoi], qubit_list[copy2][rhoi]])]
    else:
        for copy1, copy2 in combinations(range(num_copies), 2):
            for rhoi in range(rho_range):
                if decomp == "XX":
                    gate_list += controlled_swap_to_XX_gates(control,
                                                             qubit_list[copy1][rhoi],
                                                             qubit_list[copy2][rhoi])
                else:
                    gate_list += [Gate("CSWAP",
                                       target=[qubit_list[copy1][rhoi], qubit_list[copy2][rhoi]],
                                       control=control)]

    return Circuit(gate_list, n_qubits=n_qubits)


def givens_gate(target, theta, is_variational=False):
    """Generates the list of gates corresponding to a givens rotation exp(-theta*(XX+YY))

    Explicitly the two-qubit matrix is
    [[0,      0,           0,       0],
     [0,  cos(theta), -sin(theta),  0],
     [0,  sin(theta),  cos(theta),  0],
     [0,      0,            0,      0]]

    Args:
        target (list): list of two integers that indicate which qubits are involved in the givens rotation
        theta (float): the rotation angle
        is_variational (bool): Whether the rotation angle is a variational parameter.

    Returns:
        list of Gate: The list of gates corresponding to the givens rotation"""
    if len(target) != 2:
        raise ValueError("target must be a list or array of two integers")
    return [Gate("CNOT", target=target[0], control=target[1]),
            Gate("CRY", target=target[1], control=target[0], parameter=-theta, is_variational=is_variational),
            Gate("CNOT", target=target[0], control=target[1])]
