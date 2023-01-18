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

"""Module to generate the circuits necessary to implement linear combinations of unitaries
Refs:
    [1] Dominic W. Berry, Andrew M. Childs, Richard Cleve, Robin Kothari, Rolando D. Somma, "Simulating
    Hamiltonian dynamics with a truncated Taylor series" arXiv: 1412.4687 Phys. Rev. Lett. 114, 090502 (2015)
"""

import math
from typing import Union, Tuple, List

import numpy as np

from tangelo.linq import Gate, Circuit
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.operators.operators import QubitOperator, count_qubits


def get_truncated_taylor_series(qu_op: QubitOperator, kmax: int, t: float, control: Union[int, List[int]] = None) -> Circuit:
    r"""Generate Circuit to implement the truncated Taylor series algorithm as implemented in arXiv:1412.4687
    Args:
        qu_op (QubitOperator): The qubit operator to apply the truncated Taylor series exponential
        kmax (int): The maximum order of the Taylor series \exp{-1j*t*qu_op} = \sum_{k}^{kmax}(-1j*t)**k/k! qu_op**k
        t (float): The total time to evolve
        control (int or list[int]): The control qubit(s)
    Returns:
        Circuit: the circuit that implements the time-evolution of qu_op for time t with Taylor series kmax
    """

    if kmax < 1:
        raise ValueError("Taylor series can only be applied for kmax > 0")

    qu_op_size = count_qubits(qu_op)

    kprep, unitaries, rsteps = Uprepkl(qu_op, kmax, t)

    kprep_qubits = list(range(qu_op_size, qu_op_size + kprep.width))
    kprep.reindex_qubits(kprep_qubits)

    kselect = USelectkl(unitaries, qu_op_size, kmax, control)
    flip_op = sign_flip(kprep_qubits, control)

    lcu_circuit = kprep + kselect + kprep.inverse()

    amplified_lcu_circuit = lcu_circuit + flip_op + lcu_circuit.inverse() + flip_op + lcu_circuit

    # Added gates below because current implementation applies -1j*exp(-1j*H*t) and global phase
    # matters for controlled operations
    # TODO: Find a way to incorporate this phase into the time propagation natively.
    if control is not None:
        gates = [Gate("CRZ", q, control=control, parameter=np.pi/2) for q in range(qu_op_size)]
        gates += [Gate("CPHASE", q, control=control, parameter=-np.pi/2) for q in range(qu_op_size)]
        amplified_lcu_circuit += Circuit(gates)

    return amplified_lcu_circuit * rsteps


def Uprepkl(qu_op: QubitOperator, kmax: int, t: float) -> Tuple[Circuit, List[QubitOperator], int]:
    """Generate Uprep circuit using qubit encoding defined in arXiv:1412.4687
    Args:
        qu_op (QubitOperator) :: The qubit operator to obtain the Uprep circuit for
        kmax (int): the order of the truncated Taylor series
        t (float): The evolution time
    Returns:
        Circuit: The Uprep circuit for the truncated Taylor series
        list: the individual QubitOperator unitaries with only the prefactor remaining. i.e.  all coefficients are 1, -1, 1j or -1j
        int: The number of repeated applications of the circuit that need to be applied
    """

    # Incorporate sign of time into qubit operator.
    qu_op_new = np.sign(t) * qu_op
    t_new = abs(t)

    # remove the coefficient value and generate the list of QubitOperator with only its phase 1j, or -1j
    vector = list()
    unitaries = list()
    for term, coeff in qu_op_new.terms.items():
        if np.abs(coeff.imag) > 1.e-7:
            raise ValueError(f"Only real qubit operators are allowed but term {term} has coefficient {coeff}")
        if coeff.real > 0:
            unitaries.append(QubitOperator(term, -1j))
        else:
            unitaries.append(QubitOperator(term, 1j))
        vector.append(np.abs(coeff.real))

    num_terms = len(vector)
    vector = np.array(vector)
    vsum = sum(vector)

    # Calculate 1-norm of coefficients in qubit operator and obtain the maximum time-step allowed
    # These values are obtained by finding the relevant root of the kth order Taylor series polynomial approximation
    # of 2 = exp(x) (i.e. the roots of 2 = \sum_{k=0}^N x^k/k!). The limit for large k is log(2)
    poly_roots = {1: 1., 2: 0.73205081, 3: 0.69888549, 4: 0.69390315, 5: 0.69323260, 6: 0.69315552, 7: 0.69314790, 8: 0.69314724}
    max_time_step = poly_roots[kmax] / vsum if kmax < 9 else np.log(2)

    # Calculate the number of time steps required and calculate the actual 1-norm for each time-step
    time_steps = math.ceil(t_new / max_time_step)
    vsum_with_t = sum(coeff * t_new / time_steps for coeff in vector)
    expvsum = sum((vsum_with_t) ** k / math.factorial(k) for k in range(0, kmax+1))

    # Generate vector v_i = sqrt(alpha_i)/np.sqrt(1-norm) with zeros padded to create a vector of length 2**n
    vector = np.sqrt(vector) / np.sqrt(vsum)
    n_qubits = math.ceil(math.log2(num_terms))
    newvec = np.zeros(2 ** n_qubits)
    newvec[:num_terms] = vector

    # Calculate circuit that generates vector and apply controls for Taylor series
    s = StateVector(newvec, order="msq_first")
    qss = list()
    for k in range(kmax):
        qss.append(s.initializing_circuit())
        # Reindex to correct block of qubits for kth order of Taylor series
        qss[-1].reindex_qubits(list(range(kmax + k * n_qubits, kmax + (k+1) * n_qubits)))
        # Add control for the kth unary encoded qubits
        for gate in qss[-1]._gates:
            if gate.control is not None:
                gate.control += [k]
            else:
                gate.name = "C"+gate.name
                gate.control = [k]
    qstot = Circuit()
    for qs in qss:
        qstot += qs

    # Calculate coefficients for unary encoding k="0"+"0"*(kmax-k)*"1"*k where
    # first qubit encodes extra identity term needed to ensure 1-norm = 2 such that
    # oblivious amplitude amplification is applicable
    kvec = np.zeros(kmax + 2)
    for k in range(kmax+1):
        # for position "0" + "0"*(kmax-k) + "1"*k
        kvec[k] = (t_new / time_steps * vsum) ** k / math.factorial(k)
    kvec[0] += ((2 - expvsum) / 2)
    # for position "1" + "0" * kmax
    kvec[kmax+1] = ((2 - expvsum) / 2)

    kvec = np.sqrt(kvec) / np.sqrt(np.sum(kvec))

    kprep = get_unary_prep(kvec, kmax)

    # shift encoding of extra identity term to last qubit
    kprep.reindex_qubits(list(range(kmax)) + [kmax + kmax * n_qubits])

    return kprep + qstot, unitaries, time_steps


def get_unary_prep(kvec: np.ndarray, kmax: int) -> Circuit:
    """Generate the prep circuit for the unary+ancilla part of the Taylor series encoding. This implementation
    scales linearly with kmax whereas StateVector.initializing_circuit() scaled exponentially.

    Args:
        kvec (array): Array representing the coefficients needed to generate unary portion of encoding.
            Length of array is kmax+2. order "00...000", "00...001", "00...011", ..., "01...111", "1000..."
        kmax (int): The Taylor series order

    Returns:
        Circuit : The unary encoding prep circuit
    """

    # Generate ancilla value and apply "X" so control is on other portion
    # For "1" + kmax*"0"
    val = kvec[kmax + 1]
    gates = [Gate("RY", kmax, parameter=np.arcsin(val)*2), Gate("X", kmax)]

    # Keep track of remaining value in constant c
    c = np.cos(np.arcsin(val))
    for i in range(0, kmax):
        # Obtain new value to generate and rotate by np.arccos(val/c) for "0" + "0"*(kmax-i) + "1"*i
        val = kvec[i]
        control = [kmax] + [i-1] if i > 0 else [kmax]
        gates += [Gate("CRY", i, control=control, parameter=np.arccos(val/c)*2)]
        c *= np.sin(np.arccos(val/c))
    gates += [Gate("X", kmax)]

    return Circuit(gates)


def USelectkl(unitaries: List[QubitOperator], n_qubits_sv: int, kmax: int, control: Union[int, List[int]] = None) -> Circuit:
    r"""Generate the truncated Taylor series U_{Select} circuit for the list of QubitOperator as defined arXiv:1412.4687
    The returned Circuit will have qubits defined in registers |n_qubits_sv>|kmax-1>|n_qubits_u>^{kmax-1}|ancilla>
    n_qubits_sv is the number of qubits to define the state to propagate. |kmax-1> defines the unary encoding
    of the Taylor series order. kmax-1 copies of length log2(len(unitaries)) to define the binary encoding of the linear
    combination of unitaries. Finally, one ancilla qubit to ensure the success of oblivious amplitude amplification.
    Args:
        unitaries (list[QubitOperator]): The list of unitaries that defines the U_{Select} operation
        n_qubits_sv (int): The number of qubits in the statevector register
        kmax (int): The maximum Taylor series order
        control (int or list[int]): Control qubits for operation
    Returns:
        Circuit: The circuit that implements the truncated Taylor series U_{Select} operation
    """

    n_qubits_u = math.ceil(math.log2(len(unitaries)))
    if control is not None:
        control_list = control.copy() if isinstance(control, list) else [control]
    else:
        control_list = []

    gate_list = []
    for k in range(1, kmax+1):
        q_start = n_qubits_sv + kmax + (k-1) * n_qubits_u
        k_control_qubits = [n_qubits_sv + k - 1] + list(range(q_start, q_start + n_qubits_u)) + control_list

        for j, unitary in enumerate(unitaries):
            bs = bin(j).split('b')[-1]
            state_binstr = "0" * (n_qubits_u - len(bs)) + bs
            state_binstr = state_binstr[::-1]

            x_ladder = [Gate("X", q_start + q) for q, i in enumerate(state_binstr) if i == "0"]

            # Add phase to controlled-unitary
            gate_list += x_ladder
            for term, coeff in unitary.terms.items():
                phasez = np.arctan2(-coeff.imag, coeff.real)
                phasep = -phasez*2
                if abs(phasez) > 1.e-12:
                    gate_list.append(Gate("CRZ", target=0, control=k_control_qubits, parameter=phasez*2))
                if abs(phasep) > 1.e-12 or not np.isclose(abs(phasep), np.pi*2):
                    gate_list.append(Gate("CPHASE", target=0, control=k_control_qubits, parameter=phasep))
                gate_list += [Gate("C"+op, target=index, control=k_control_qubits) for index, op in term]
            gate_list += x_ladder

    # Add -I term to ensure total sum equals 2 for oblivious amplitude amplification
    k_control_qubits = [n_qubits_sv+ki for ki in range(kmax)] + [n_qubits_sv + kmax + kmax*n_qubits_u] + control_list
    x_ladder = [Gate("X", n_qubits_sv + q) for q in range(kmax)]
    gate_list += x_ladder
    gate_list.append(Gate("CRZ", target=0, control=k_control_qubits, parameter=2*np.pi))
    gate_list += x_ladder

    return Circuit(gate_list)


def sign_flip(qubit_list: List[int], control: Union[int, List[int]] = None) -> Circuit:
    """Generate Circuit corresponding to the sign flip of the |0>^n vector for the given qubit_list
    Args:
        qubit_list (list[int]): The list of n qubits for which the 2*|0>^n-I operation is generated
        control (int or list[int]): Control qubit or list of control qubits.
    Returns:
        Circuit: The circuit that generates the sign flip on |0>^n
    """
    if control is not None:
        fcontrol_list = control.copy() if isinstance(control, list) else [control]
    else:
        fcontrol_list = []
    gate_list = []

    x_ladder = [Gate("X", q) for q in qubit_list]
    fcontrol_list += qubit_list[:-1]

    gate_list += x_ladder
    gate_list.append(Gate('H', target=qubit_list[-1]))
    gate_list.append(Gate('CX', target=qubit_list[-1], control=fcontrol_list))
    gate_list.append(Gate('H', target=qubit_list[-1]))
    gate_list += x_ladder
    return Circuit(gate_list)


def get_oaa_lcu_circuit(qu_op: QubitOperator, control: Union[int, List[int]] = None) -> Circuit:
    """Apply qu_op using linear combination of unitaries (LCU) with oblivious amplitude amplification (OAA)
    1-norm of coefficients must be less than 2. The unitarity of qu_op is not checked by the algorithm
    Args:
        qu_op (QubitOperator): The qu_op to apply. Must be nearly unitary for algorithm to succeed with high probability.
        control (int or list[int]): Control qubit(s)
    Returns:
        Circuit: The circuit that implements the linear combination of unitaries for the qu_op
    """

    uprep, uselect, qu_op_qubits, uprep_qubits, _ = get_uprep_uselect(qu_op, control, make_alpha_eq_2=True)

    flip_op = sign_flip(uprep_qubits, control=control)
    w = uprep + uselect + uprep.inverse()

    amplified_lcu_circuit = w + flip_op + w.inverse() + flip_op + w

    # Added gates below because current implementation applies -1j*exp(-1j*H*t) and global phase
    # matters for controlled operations
    # TODO: Find a way to incorporate this phase into the time propagation natively.
    if control is not None:
        gates = [Gate("CRZ", q, control=control, parameter=np.pi) for q in qu_op_qubits]
        gates += [Gate("CPHASE", q, control=control, parameter=-np.pi) for q in qu_op_qubits]
        amplified_lcu_circuit += Circuit(gates)

    return amplified_lcu_circuit


def get_uprep_uselect(qu_op: QubitOperator, control: Union[int, List[int]] = None, make_alpha_eq_2: bool = False
                      ) -> Tuple[Circuit, Circuit, List[int], List[int], float]:
    """Get uprep and (controlled-)uselect circuits along with their corresponding qubits for a QubitOperator.
    Args:
        qu_op (QubitOperator): The qu_op to apply.
        control (int or list[int]): Control qubit(s).
        make_alpha_eq_2: Make 1-norm equal 2 by adding and subtracting identity terms. Useful for oblivious amplitude amplification
    Returns:
        Circuit: Uprep circuit
        Circuit: Uselect circuit
        List[int]: QubitOperator qubits
        List[int]: Auxillary qubits
        float: alpha = 1-norm of coefficients of applied qu_op
    """

    if control is not None:
        control_list = control.copy() if isinstance(control, list) else [control]
    else:
        control_list = []

    unitaries = list()
    vector = list()
    max_qu_op = count_qubits(qu_op)
    for term, coeff in qu_op.terms.items():
        acoeff = np.abs(coeff)
        if acoeff > 1.e-8:
            vector += [acoeff]
            unitaries += [QubitOperator(term, -coeff / acoeff)]
        else:
            vector += [0]
            unitaries += [QubitOperator((), 1)]

    # create U_{prep} from sqrt of coefficients
    vector = np.array(vector)
    alpha = sum(vector)

    if make_alpha_eq_2:
        # Check that 1-norm is less than 2 and add and subtract terms proportional to the identity to obtain 1-norm equal to 2
        if alpha > 2 + 1.e-10:
            raise ValueError("Can not make qu_op 1-norm equal 2 as it is already greater than 2")
        vector = np.concatenate((vector, np.array([max((2 - alpha)/2, 0), max((2 - alpha)/2, 0)])))
        unitaries += [QubitOperator(tuple(), 1), QubitOperator(tuple(), -1)]
        alpha = 2

    num_terms = len(vector)
    vector = np.sqrt(vector)
    vector = vector / np.linalg.norm(vector)
    n_qubits = math.ceil(math.log2(num_terms))
    newvec = np.zeros(2**n_qubits)
    newvec[0:num_terms] = vector[:]
    s = StateVector(newvec, order="lsq_first")
    uprep = s.initializing_circuit()
    uprep_qubits = list(range(max_qu_op, max_qu_op + n_qubits))
    uprep.reindex_qubits(uprep_qubits)

    # Generate U_{Select}
    gate_list = list()
    control_qubits = uprep_qubits + control_list
    for j, unitary in enumerate(unitaries):
        bs = bin(j).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs
        x_ladder = [Gate("X", q + max_qu_op) for q, i in enumerate(state_binstr) if i == "0"]
        gate_list += x_ladder
        for term, coeff in unitary.terms.items():
            phasez = np.arctan2(-coeff.imag, coeff.real)
            phasep = -phasez*2

            if abs(phasez) > 1.e-12:
                gate_list.append(Gate("CRZ", target=0, control=control_qubits, parameter=phasez*2))
            if abs(phasep) > 1.e-12 or not np.isclose(abs(phasep), np.pi*2):
                gate_list.append(Gate("CPHASE", target=0, control=control_qubits, parameter=phasep))
            gate_list += [Gate("C"+op, target=index, control=control_qubits) for index, op in term]
        gate_list += x_ladder

    uselect = Circuit(gate_list)

    return uprep, uselect, list(range(max_qu_op)), uprep_qubits, alpha


def get_lcu_qubit_op_info(qu_op: QubitOperator()) -> Tuple[List[int], List[int], float]:
    """Return the operator and auxillary qubit indices and 1-norm for the LCU block decomposition of given QubitOperator.
    Args:
        qu_op (QubitOperator): The qubit operator to decompose into an Linear Combination of Unitaries block encoding.

    Returns:
        List[int]: The qubit operator indices.
        List[int]: The auxillary qubits for the uprep circuit.
        float: The 1-norm of the qubit operator.
    """

    max_qu_op = count_qubits(qu_op)
    num_terms = len(qu_op.terms)
    alpha = sum([abs(v) for v in qu_op.terms.values()])
    n_qubits = math.ceil(math.log2(num_terms))

    return list(range(max_qu_op)), list(range(max_qu_op, max_qu_op + n_qubits)), alpha
