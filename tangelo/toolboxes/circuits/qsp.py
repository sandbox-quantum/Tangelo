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

"""Module to generate the circuits necessary to implement quantum signal processing (QSP) Hamiltonian Simulation

Ref:
[1]: Yulong Dong, Xiang Meng, K. Birgitta Whaley, Lin Lin, "Efficient phase-factor evaluation in quantum signal processing"
2021, arXiv:2002.11649 (https://arxiv.org/abs/2002.11649)
"""

import math
from typing import Union, List, Tuple

import numpy as np

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.operators import QubitOperator
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.circuits.lcu import get_uprep_uselect, sign_flip, get_lcu_qubit_op_info


def ham_sim_phases(tau: float, eps: float = 1.e-2, n_attempts: int = 10, method: str = "laurent") -> Tuple[List[float], List[float]]:
    """Generate the phases required for QSP time-evolution using the pyqsp package

    Args:
        tau (float): The evolution time.
        eps (float): The precision to calculate the factors. Higher precision is more likely to fail.
        n_attempts (int): The number of attempts to calculate the phase factors. Often multiple tries are required.
        method: "laurent": Laurent Polynomial method (unstable but fast). "tf" requires TensorFlow installed and is stable but very slow.

    Returns:
        List[float]: The phases for Cos(Ht).
        List[float]: The phases for i*Sin(Ht).
    """
    try:
        import pyqsp
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pyqsp package is required to calculate QSP time-evolution phases using 'laurent' or 'tf' method.")

    from pyqsp import angle_sequence
    from pyqsp.angle_sequence import AngleFindingError
    from pyqsp.completion import CompletionError

    # Compute phases for real part Cos(Ht) of Exp(iHt)
    pg = pyqsp.poly.PolyCosineTX()
    pcoefs, _ = pg.generate(tau=tau,
                            return_coef=True,
                            ensure_bounded=True,
                            return_scale=True, epsilon=eps)
    for i in range(n_attempts):
        try:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
                pcoefs, eps=eps, suc=1-eps/10, method=method)
        except (AngleFindingError, CompletionError):
            if i == n_attempts-1:
                raise RuntimeError("Real phases calculation failed, increase n_attempts or eps")
            else:
                print(f"Attempt {i+2} for the real coefficients")
        else:
            break

    # Compute phases for imaginary part i*Sin(Ht) of Exp(iHt)
    pg = pyqsp.poly.PolySineTX()
    pcoefs, _ = pg.generate(tau=tau,
                            return_coef=True,
                            ensure_bounded=True,
                            return_scale=True, epsilon=eps)
    for i in range(n_attempts):
        try:
            phiset2 = angle_sequence.QuantumSignalProcessingPhases(
                pcoefs, eps=eps, suc=1-eps/10, method=method)
        except (AngleFindingError, CompletionError):
            if i == n_attempts-1:
                raise RuntimeError("Imaginary phases calculation failed, increase n_attempts or eps")
            else:
                print(f"Attempt {i+2} for the imaginary phases")
        else:
            break
    return phiset, phiset2


def ham_sim_phases_QSPPACK(folder: str, tau: float, eps: float = 1.e-7) -> Tuple[List[float], List[float]]:
    """Generate the phases required for QSP time-evolution using the QSPPACK package.

    QSPPACK is an Matlab/Octave based package that calculates the phase factors. To run this function, a user needs to have:
    1) Octave installed and accessible in path. Octave can be found at https://octave.org/
    2) oct2py installed. pip install oct2py
    3) QSPPACK downloaded in an accessible folder. Can be found at https://github.com/qsppack/QSPPACK/tree/dev

    Args:
        folder (str): The folder location of QSPPACK.
        tau (float): The evolution time.
        eps (float): The precision to calculate the factors.

    Returns:
        List[float]: The phases for Cos(Ht).
        List[float]: The phases for i*Sin(Ht).
    """
    from oct2py import octave
    from scipy.special import jn

    octave.addpath(folder)
    opts = octave.struct("criteria", eps)
    maxorder = math.ceil(1.4*tau+np.log(1e14))
    if maxorder % 2 == 1:
        maxorder -= 1

    coef = np.zeros((maxorder//2 + 1, 1), dtype=float, order='F')
    for i in range(1, len(coef)+1):
        coef[i-1][0] = (-1)**(i-1)*jn(2*(i-1), tau)
    coef[0] = coef[0]/2
    phi1, _ = octave.QSP_solver(coef, 0, opts, nout=2)

    coef = np.zeros((maxorder//2+1, 1), dtype=float, order='F')
    for i in range(1, len(coef)+1):
        coef[i-1][0] = (-1)**(i-1)*jn(2*i-1, tau)
    phi2, _ = octave.QSP_solver(coef, 1, opts, nout=2)

    return list(phi1.flatten()), list(phi2.flatten())


def zero_controlled_cnot(qubit_list: List[int], target: int, control: Union[int, List[int]]) -> Circuit:
    """Return the circuit for a zero-controlled CNOT gate with possible extra controls on 1.

    Args:
        qubit_list (List[int]): The qubits controlled by zero.
        target (int): The target qubit.
        control (List[int] or int): The qubits controlled by one.

    Returns:
        Circuit: The zero controlled cnot circuit with possible extra controls on 1.
    """
    control_list = control if isinstance(control, list) else [control]
    x_ladder = [Gate("X", q) for q in qubit_list]
    gates = x_ladder + [Gate("CX", target=target, control=qubit_list+control_list)] + x_ladder
    return Circuit(gates)


def get_qsp_circuit_no_anc(cua: Circuit, m_qs: List[int], angles: List[float], control: Union[int, List[int]] = None, with_z: bool = False) -> Circuit:
    """Generate the ancilla free QSP circuit as defined in Fig 16 of https://arxiv.org/abs/2002.11649.

    Args:
        cua (Circuit): The controlled unitary (U_A).
        m_qs (List[int]): The m ancilla qubits used for the Uprep circuit.
        angles (List[float]): The phases for the QSP circuit.
        control (Union[int, List[int]]): The control qubit(s).
        with_z (bool): If True, an extra CZ gate is included. This adds a 1j phase to the operation.

    Returns:
        Circuit: The ancilla free QSP circuit
    """
    c_list = list()
    if control is not None:
        c_list += control if isinstance(control, list) else [control]

    anc = m_qs[-1]+1
    if with_z:
        qubcirc = Circuit([Gate("CH", anc, control=c_list), Gate("CZ", anc, control=c_list)])
    else:
        qubcirc = Circuit([Gate("CH", anc, control=c_list)])

    circ_angles = [angles[0]+np.pi/4]
    for ang in angles[1:-1]:
        circ_angles += [ang + np.pi/2]
    circ_angles += [angles[-1]+np.pi/4]

    zcnot = zero_controlled_cnot(m_qs, [anc], [])

    qubcirc += zcnot + Circuit([Gate("CRZ", target=anc, parameter=2*circ_angles[-1], control=c_list)]) + zcnot + cua
    for j, ang in enumerate(circ_angles[-2:0:-1]):
        qubcirc += Circuit([Gate("CZ", anc, control=c_list)]) + zcnot + Circuit([Gate("CRZ", target=anc, parameter=2*ang, control=c_list)]) + zcnot + cua
    qubcirc += Circuit([Gate("CZ", anc, control=c_list)]) + zcnot + Circuit([Gate("CRZ", target=anc, parameter=2*circ_angles[0], control=c_list)]) + zcnot

    qubcirc += Circuit([Gate("CH", anc, control=c_list)])

    return qubcirc


def get_qsp_hamiltonian_simulation_qubit_list(qu_op: QubitOperator) -> List[int]:
    "Returns the list of qubits used for the QSP Hamiltonian simulation algorithm"
    qu_op_qs, uprep_qs, _ = get_lcu_qubit_op_info(qu_op)
    return qu_op_qs + uprep_qs + list(range(uprep_qs[-1]+1, uprep_qs[-1]+4))


def get_qsp_hamiltonian_simulation_circuit(qu_op: QubitOperator, tau: float, eps: float = 1.e-4, control: Union[int, List[int]] = None, n_attempts: int = 10,
                                           method: str = 'laurent', folder: str = None) -> Circuit:
    """Returns Quantum Signal Processing (QSP) Hamiltonian simulation circuit for a given QubitOperator for time tau.

    The circuits are derived from https://arxiv.org/abs/2002.11649.
    The list of qubits used for the circuit can be found by qubit_list = get_qs_hamiltonian_simulation_qubit_list(qu_op)
    qu_op must have it's spectral range in [-1, 1]. This property is not checked by this function.

    Args:
        qu_op (QubitOperator): The qubit operator defining the QSP time-evolution circuit.
        tau (float): The evolution time.
        eps (float): The allowed error of the time-evolution. Higher accuracy requires longer circuits, and obtaining the phases can be unstable.
        control (Union[int, List[int]]): The control qubit(s).
        n_attempts (int): The number of attempts to calculate the phase factors using pyqsp.
        method (str): "laurent": Use laurent polynomial method of pyqsp, "tf": Use TensorFlow with pyqsp, "QSPPACK": Use QSPPACK to calculate phases
        folder (str): Folder that contains QSPPACK.

    Returns:
        Circuit: The QSP Hamiltonian simulation circuit with oblivious amplitude amplification to ensure very high success probability.
    """

    if control is not None:
        control_list = control.copy() if isinstance(control, list) else [control]
    else:
        control_list = []

    # If tau is negative, flip sign of tau and qu_op.
    flip_tau_sign = (tau < 0.)
    if flip_tau_sign:
        qu_op = -qu_op
        tau = -tau

    qu_op_qs, m_qs, alpha = get_lcu_qubit_op_info(qu_op)

    if method.lower() in ["laurent", "tf"]:
        anglesr, anglesi = ham_sim_phases(tau*alpha, eps, n_attempts, method)
    elif method.lower() == "qsppack":
        anglesr, anglesi = ham_sim_phases_QSPPACK(folder, tau*alpha, eps)
    else:
        raise ValueError(f"{method} is not a valid keyword to calculate phases. Must be Laurent, TF, or QSPPACK")

    # Want 1-norm of coefficents to sum to 1/np.sin(np.pi/(2*(2*3+1))) so three oblivious amplitude amplifications results
    # in success probability of 1. |(cos(Ht)|=2 and |iSin(Ht))|=2 so need to add (tsum-4)/2 I - (tsum-4)/2 I.
    tsum = 1/np.sin(np.pi/(2*(2*3+1)))
    v = [(tsum-4)/2, (tsum-4)/2, 2, 2]
    v = np.sqrt(np.array(v))/np.sqrt(tsum)

    # Leave gap of 1-qubit for qsp_circuit and list LCU qubits to add cos(Ht) + Sin(Ht) + (tsum-4)/2 I - (tsum-4)/2 I
    lcu_qs = list(range(m_qs[-1]+2, m_qs[-1]+4))
    flip_qs = m_qs + [m_qs[-1]+1] + lcu_qs

    uprep, uselect, qu_op_qs, m_qs, alpha = get_uprep_uselect(qu_op, control=lcu_qs+control_list)
    cua = uprep + uselect + uprep.inverse()

    s = StateVector(v, order="msq_first")
    uprep = s.initializing_circuit()
    uprep.reindex_qubits([lcu_qs[0], lcu_qs[1]])

    circ = uprep + Circuit([Gate("X", lcu_qs[0])])
    # real part cos(Ht)
    circ += get_qsp_circuit_no_anc(cua, m_qs, anglesr, control=lcu_qs+control_list, with_z=False)
    circ += Circuit([Gate("X", lcu_qs[0])])
    # imaginary part i*sin(Ht)
    circ += get_qsp_circuit_no_anc(cua, m_qs, anglesi, control=lcu_qs+control_list, with_z=False)
    # -I to ensure probability is np.arcsin
    circ += Circuit([Gate("X", lcu_qs[1]), Gate("CRZ", target=0, parameter=2*np.pi, control=lcu_qs+control_list), Gate("X", lcu_qs[1])])

    circ += uprep.inverse()

    if control is not None:
        gates = [Gate("CRZ", qu_op_qs[0], control=control, parameter=2*np.pi)] if len(anglesi) % 4 == 2 else []
    else:
        gates = [Gate("RZ", qu_op_qs[0], parameter=2*np.pi)] if len(anglesi) % 4 == 2 else []

    if flip_tau_sign:
        qu_op = -qu_op
        tau = -tau

    return circ + (sign_flip(flip_qs) + circ.inverse() + sign_flip(flip_qs) + circ)*3 + Circuit(gates)
