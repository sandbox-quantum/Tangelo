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

"""Provide useful functions, corresponding to common patterns in quantum
chemistry circuits (CNOT ladders, Pauli-word to circuit translation ...) to
facilitate the assembly of ansatz quantum circuits.
"""

from copy import deepcopy

import numpy as np
from openfermion.ops import FermionOperator as ofFermionOperator
from openfermion.ops import InteractionOperator as ofInteractionOperator
from openfermion.ops import QubitOperator as ofQubitOperator

from qsdk.backendbuddy import Circuit, Gate
from qsdk.toolboxes.operators import FermionOperator, QubitOperator
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping, get_fermion_operator


def pauli_op_to_gate(index, op, inverse=False):
    """Return the change-of-basis gates required to map pauli words to quantum
    circuit as per Whitfield 2010 (https://arxiv.org/pdf/1001.3855.pdf).
    """
    if op == "X":
        return Gate("H", index)
    elif op == "Y":
        angle = 0.5*np.pi
        return Gate("RX", index, parameter=angle) if not inverse else Gate("RX", index, parameter=-angle+4*np.pi)


def pauliword_to_circuit(pauli_word, coef, variational=True, control=None):
    """Generate a list of Gate objects corresponding to the exponential of a pauli word.
    The process is described in Whitfield 2010 https://arxiv.org/pdf/1001.3855.pdf

    Args:
        pauli_word (): The pauli_word to exponentiate
        coef (float): The coefficient in the exponentiation
        variational (bool): When creating the Gate objects, label the (controlled-)Rz gate as variational
        control (integer): The control qubit label

    Returns:
        list: list of Gate objects that represent the exponentiation.
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


def circuit_for_exponentiated_qubit_operator(qubit_op, time=1., variational=False, trotter_order=1, control=None):
    """Generate the exponentiation of a qubit operator in first_order Trotterized form.
    The algorithm is described in Whitfield 2010 https://arxiv.org/pdf/1001.3855.pdf

    Args:
        qubit_op  (QubitOperator):  qubit hamiltonian to exponentiate
        time (float): the coefficient to multiple to each coef for the exponential of each term
        variational (bool) : Whether the coefficients are variational
        trotter_order (int): order of trotter approximation

    Returns:
        Circuit: circuit corresponding to exponentiation of qubit operator
    """
    pauli_words = qubit_op.terms.items()
    num_ops = len(pauli_words)

    if isinstance(time, float):
        evolve_time = np.ones((num_ops,), dtype=np.double) * time
    else:
        if len(time) == num_ops:
            evolve_time = np.array(time)

    if trotter_order == 2:
        evolve_time /= trotter_order

    phase = 1.
    exp_pauli_word_gates = list()
    for i, (pauli_word, coef) in enumerate(pauli_words):
        if (len(pauli_word) > 0):  # identity terms do not contribute to evolution outside of a phase
            if abs(np.real(coef)*evolve_time[i]) > 1.e-10:
                exp_pauli_word_gates += pauliword_to_circuit(pauli_word,
                                                             np.real(coef)*evolve_time[i],
                                                             variational=variational,
                                                             control=control)
        else:
            if control is None:
                phase *= np.exp(-1j * coef * evolve_time[i])
            else:
                exp_pauli_word_gates += [Gate("PHASE", target=control, parameter=-np.real(coef)*evolve_time[i])]

    if trotter_order == 2:
        exp_pauli_word_gates += [exp_pauli_word for exp_pauli_word in reversed(exp_pauli_word_gates)]
        phase *= phase
    return Circuit(exp_pauli_word_gates), phase


def trotterize(operator, time=1., num_trotter_steps=1, trotter_order=1, variational=False,
               mapping_options=dict(), control=None):
    """Generate the circuit that represents time evolution of an operator.
    This circuit is generated as a trotterization of a qubit operator which is either the input
    or mapped from the given fermion operator.

    Args:
        operator  (QubitOperator or FermionOperator):  operator to time evolve
        time (float or array): The time to evolve the whole system or individiual times for each
            term in the operator. If an array, must match the number of terms
            in operator
        variational (bool): whether the coefficients are variational
        trotter_order (int): order of trotter approximation, 1 or 2 supported
        num_trotter_steps (int): The number of different time steps taken for total time t
        mapping_options (dict): Defines the desired Fermion->Qubit mapping
                                "up_then_down": False
                                "qubit_mapping": 'jw'
                                "n_spinorbitals": None
                                "n_electrons": None

    Returns:
        Circuit: circuit corresponding to time evolution of the operator
    """
    if isinstance(operator, (FermionOperator, ofFermionOperator, ofInteractionOperator)):
        options = {"up_then_down": False,
                   "qubit_mapping": "jw",
                   "n_spinorbitals": None,
                   "n_electrons": None}
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in mapping_options.items():
            if k in options:
                options[k] = v
            else:
                raise KeyError(f"Keyword :: {k}, not a valid fermion to qubit mapping option")
        if isinstance(operator, ofInteractionOperator):
            operator = get_fermion_operator(operator)
        if isinstance(time, float):
            evolve_time = np.ones(len(operator.terms)) * time
        elif isinstance(time, list) or isinstance(time, np.ndarray):
            if len(time) == len(operator.terms):
                evolve_time = np.array(time)
            else:
                raise ValueError(f'time as length {len(time)} but FermionicOperator has length {len(operator.terms)}')
        else:
            raise ValueError("time must be a float or array")
        new_operator = FermionOperator()
        for i, term in enumerate(operator.terms):
            new_operator += FermionOperator(term, operator.terms[term]*evolve_time[i]/num_trotter_steps)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=new_operator,
                                            mapping=options["qubit_mapping"],
                                            n_spinorbitals=options["n_spinorbitals"],
                                            n_electrons=options["n_electrons"],
                                            up_then_down=options["up_then_down"])
        circuit, phase = circuit_for_exponentiated_qubit_operator(qubit_op,
                                                                  time=1.,  # time is already included
                                                                  trotter_order=trotter_order,
                                                                  variational=variational,
                                                                  control=control)

    elif isinstance(operator, (QubitOperator, ofQubitOperator)):
        qubit_op = deepcopy(operator)
        if isinstance(time, float):
            evolve_time = time / num_trotter_steps
        elif isinstance(time, np.ndarray) or isinstance(time, list):
            if len(time) == len(operator.terms):
                evolve_time = np.array(time) / num_trotter_steps
            else:
                raise ValueError(f"time as length {len(time)} but FermionicOperator has length {len(operator.terms)}")
        circuit, phase = circuit_for_exponentiated_qubit_operator(qubit_op,
                                                                  time=evolve_time,
                                                                  trotter_order=trotter_order,
                                                                  variational=variational,
                                                                  control=control)
    else:
        raise ValueError("Only FermionOperator or QubitOperator allowed")

    if num_trotter_steps == 1:
        return circuit, phase
    else:
        final_circuit = deepcopy(circuit)
        final_phase = deepcopy(phase)
        for i in range(1, num_trotter_steps):
            final_circuit += circuit
            final_phase *= phase
        return final_circuit, final_phase


def qft_rotations(gate_list, qubit_list, prefac=1):
    '''Returns the list of gates required for a quantum fourier transform.

    Args:
        gate_list (list): List of Gate elements
        qubit_list (list): List of integers for which the qft operations are performed

    Returns:
        list: List of gates for rotation portion of qft circuit'''
    n = len(qubit_list)
    if n == 0:
        return gate_list
    n -= 1
    gate_list += [Gate('H', target=qubit_list[n])]
    for i, qubit in enumerate(qubit_list[:n]):
        gate_list += [Gate("CPHASE", control=qubit, target=qubit_list[n], parameter=prefac*np.pi/2**(n-i))]

    qft_rotations(gate_list, qubit_list[:n], prefac=prefac)


def swap_registers(gate_list, qubit_list):
    '''Function to swap register order.
    Args:
        gate_list (list): List of Gate
        qubit_list (list): List of integers for the locations of the qubits

    Result:
        list: The operations that swap the register order'''
    n = len(qubit_list)
    for qubit_index in range(n//2):
        gate_list += [Gate("SWAP", target=[qubit_list[qubit_index], qubit_list[n - qubit_index - 1]])]
    return gate_list


def qft_circuit(qubits, n_qubits_in_circuit=None, inverse=False):
    """Returns the QFT or iQFT circuit given a list of qubits to act on.

    Args:
        qubits (int or list): The list of qubits to apply the QFT circuit to. If an integer.
            the operation is applied to the [0,...,qubits-1] qubits
        n_qubits_in_circuit: Argument to initialize a Circuit with the desired number of qubits.
        inverse (bool): If True, the inverse QFT is applied. If False, QFT is applied

        Returns:
            Circuit: The circuit that applies QFT or iQFT to qubits
        """
    if isinstance(qubits, int):
        qubit_list = [i for i in range(qubits)]
    elif isinstance(qubits, list):
        qubit_list = qubits
    else:
        raise KeyError('qubits must be an int or list of ints')

    swap_gates = list()
    swap_registers(swap_gates, qubit_list)

    qft_gates = list()
    if inverse:
        qft_rotations(qft_gates, qubit_list, prefac=-1)
        qft_gates = [gate for gate in reversed(qft_gates)]
        qft_gates = swap_gates + qft_gates
    else:
        qft_rotations(qft_gates, qubit_list)
        qft_gates += swap_gates

    return Circuit(qft_gates, n_qubits=n_qubits_in_circuit)


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
        gates = []
        for index, op in pauli_word:
            gates += [Gate(name='C'+op, target=index, control=control)]
        pauliword_circuits.append(Circuit(gates, n_qubits=n_qubits))
    return pauliword_circuits


def decomp_controlled_swap_crx(c, n1, n2):
    '''Exact decomposition of controlled swap into 1- and 2-qubit gates

    Args:
        c (int): control qubit
        n1 (int): first target qubit
        n2 (int): second target qubit

    Returns:
        list: List of Gate that applies controlled swap operation
    '''
    gates = [Gate('RY', target=n1, parameter=np.pi/2),
             Gate('RZ', target=n2, parameter=5*np.pi/2),
             Gate('CRX', control=n1, target=n2, parameter=np.pi),
             Gate('RZ', target=n2, parameter=7*np.pi/2),
             Gate('CRX', control=c, target=n2, parameter=5*np.pi/2),
             Gate('CRX', control=c, target=n1, parameter=np.pi/2),
             Gate('CRX', control=n1, target=n2, parameter=np.pi),
             Gate('RZ', target=n1, parameter=3*np.pi/2),
             Gate('CRX', control=c, target=n1, parameter=3*np.pi/2),
             Gate('RY', target=n1, parameter=7*np.pi/2),
             Gate('RZ', target=n2, parameter=3*np.pi/2),
             Gate('CRX', control=n2, target=n1, parameter=np.pi),
             Gate('RZ', target=c, parameter=7*np.pi/2)]
    return gates


def decomp_controlled_swap_crx_ue(c, n1, n2):
    '''Unitary equivalent decomposition of controlled swap into 1- and 2-qubit gates

    Args:
        c (int): control qubit
        n1 (int): first target qubit
        n2 (int): second target qubit

    Returns:
        list: List of Gate that applies controlled swap operation
    '''
    gates = [Gate('CRX', control=c, target=n1, parameter=3*np.pi),
             Gate('CRX', control=n1, target=n2, parameter=np.pi),
             Gate('RZ', target=n1, parameter=5*np.pi/2),
             Gate('RZ', target=n2, parameter=13*np.pi/4),
             Gate('CRX', control=n2, target=n1, parameter=np.pi/2),
             Gate('CRX', control=c, target=n2, parameter=3*np.pi),
             Gate('CRX', control=c, target=n1, parameter=np.pi/2),
             Gate('RZ', target=c, parameter=np.pi/4)]
    return gates


def decomp_controlled_swap_xx_ue(c, n1, n2):
    '''Unitary equivalent decomposition of controlled swap into 1- and xx 2-qubit gate

    Args:
        c (int): control qubit
        n1 (int): first target qubit
        n2 (int): second target qubit

    Returns:
        list: List of Gate that applies controlled swap operation
    '''
    gates = [Gate('RY', target=c, parameter=7*np.pi/2.),
             Gate('RZ', target=n2, parameter=np.pi/2.),
             Gate('XX', target=[n1, n2], parameter=7*np.pi/2.),
             Gate('XX', target=[c, n2], parameter=7*np.pi/2.),
             Gate('RY', target=n1, parameter=5*np.pi/4),
             Gate('RY', target=c, parameter=5*np.pi/2),
             Gate('XX', target=[n1, n2], parameter=3*np.pi/2),
             Gate('RZ', target=n1, parameter=3*np.pi/4),
             Gate('XX', target=[c, n2], parameter=11*np.pi/4),
             Gate('XX', target=[c, n1], parameter=np.pi/2),
             Gate('RY', target=c, parameter=np.pi/2),
             Gate('RZ', target=c, parameter=5*np.pi/4)]
    return gates


def decomp_controlled_swap_xx(c, n1, n2):
    '''Unitary equivalent decomposition of controlled swap into 1- and xx 2-qubit gate

    Args:
        c (int): control qubit
        n1 (int): first target qubit
        n2 (int): second target qubit

    Returns:
        list: List of Gate that applies controlled swap operation
    '''
    gates = [Gate('RY', target=c, parameter=7*np.pi/2.),
             Gate('RZ', target=n1, parameter=7*np.pi/2.),
             Gate('XX', target=[n1, n2], parameter=5*np.pi/2.),
             Gate('RZ', target=n1, parameter=7*np.pi/4.),
             Gate('RZ', target=n2, parameter=3*np.pi/4.),
             Gate('RY', target=n1, parameter=np.pi/2.),
             Gate('XX', target=[c, n2], parameter=7*np.pi/2.),
             Gate('RY', target=n2, parameter=11*np.pi/4),
             Gate('XX', target=[n1, n2], parameter=7*np.pi/2.),
             Gate('XX', target=[c, n2], parameter=np.pi/4.),
             Gate('RZ', target=n2, parameter=np.pi/4),
             Gate('XX', target=[c, n2], parameter=5*np.pi/2),
             Gate('RY', target=c, parameter=5*np.pi/2),
             Gate('RZ', target=n1, parameter=5*np.pi/2),
             Gate('RY', target=n2, parameter=7*np.pi/4),
             Gate('XX', target=[n1, n2], parameter=7*np.pi/2),
             Gate('RY', target=n1, parameter=np.pi/2),
             Gate('RZ', target=c, parameter=11*np.pi/4)]
    return gates


def derangement_circuit(qubit_list, control=None, n_qubits=None, decomp=None):
    """returns the derangement circuit for multiple copies of a state

    Args:
        qubit_list (list of list(int)): Each item in the list is a list of qubit registers for
                                        each copy. The length of each list of qubit registers
                                        must be the same.
        control (int): The control register to be measured.
        n_qubits (int): The number of qubits in the circuit.
        decomp (str): Use the decomposed controlled-swap into 1- and 2-qubit gates.
                      "crx": 2-qubit gate is controlled rx, exact decomposition
                      "crxUE": 2-qubit gate is controlled rx, unitary equivalent
                      "xxUE": 2-qubit gate is xx, unitary equivalent

    Returns:
        Circuit: The derangement circuit
    """
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
        for copy1 in range(num_copies):
            for copy2 in range(copy1+1, num_copies):
                for rhoi in range(rho_range):
                    gate_list += [Gate('SWAP', target=[qubit_list[copy1][rhoi], qubit_list[copy2][rhoi]])]
        if n_qubits is None:
            n_qubits = rho_range * num_copies
    else:
        for copy1 in range(num_copies):
            for copy2 in range(copy1+1, num_copies):
                for rhoi in range(rho_range):
                    if decomp is not None:
                        if decomp == 'crxUE':
                            gate_list += decomp_controlled_swap_crx_ue(control,
                                                                       qubit_list[copy1][rhoi],
                                                                       qubit_list[copy2][rhoi])
                        elif decomp == 'crx':
                            gate_list += decomp_controlled_swap_crx(control,
                                                                    qubit_list[copy1][rhoi],
                                                                    qubit_list[copy2][rhoi])
                        elif decomp == 'xxUE':
                            gate_list += decomp_controlled_swap_xx_ue(control,
                                                                      qubit_list[copy1][rhoi],
                                                                      qubit_list[copy2][rhoi])
                        elif decomp == 'xx':
                            gate_list += decomp_controlled_swap_xx(control,
                                                                   qubit_list[copy1][rhoi],
                                                                   qubit_list[copy2][rhoi])
                        else:
                            raise ValueError(f"{decomp} is not a valid controlled swap decomposition")
                    else:
                        gate_list += [Gate('CSWAP',
                                           target=[qubit_list[copy1][rhoi], qubit_list[copy2][rhoi]],
                                           control=control)]

        if n_qubits is None:
            n_qubits = rho_range * num_copies + 1

    return Circuit(gate_list, n_qubits=n_qubits)
