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

"""
    Modules for analytically evaluating the expectation value of a QubitOperator
    with a qubit mean field (QMF) wave function and purifying a QMF wave function.
"""

import numpy as np
from random import choice
from cmath import sin, cos
from functools import reduce

from tangelo.backendbuddy import Circuit, Gate
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector

def calc_op_expval(args):
    """
    Analytical evaluation of the expectation value of a QubitOperator with a QMF wave function.
    <QMF| X | QMF> = cos(phi) * sin(theta)
    <QMF| Y | QMF> = sin(phi) * sin(theta)
    <QMF| Z | QMF> = cos(theta)
    For more information, see arXiv:1901.04715v1.

    Args:
        args (tuple): a tuple containing a single qubit operator term and its coefficient.
    Returns:
        val (float or complex float): expectation value
    """

    op, op_avg, qmf_var_params = args 
    for term in op:
        idx, pauli = term
        theta, phi = qmf_var_params[idx], qmf_var_params[idx + qmf_var_params.size // 2]
        if pauli == "X":
            op_avg *= cos(phi) * sin(theta)
        elif pauli == "Y":
            op_avg *= sin(phi) * sin(theta)
        elif pauli == "Z":
            op_avg *= cos(theta)
    return op_avg

def get_op_expval(qubit_op, qmf_var_params):
    """
    Driver for evaluation of the expectation value of a QubitOperator with a QMF wave function.

    Args:
        qubit_op (QubitOperator): qubit operator for which to evaluate <QMF| operator |QMF>
        qmf_var_params (numpy array of floats): QMF variational parameter set {Omega}.
    Returns:
        val (complex float): expectation value
    """

    def op_data():
        for term, coeff in qubit_op.terms.items():
            yield term, coeff, qmf_var_params
    val = list(map(calc_op_expval, op_data()))
    val = reduce(lambda x, y:x+y, val) if val else complex(0.)
    return val

def initialize_qmf_state_from_hf_vec(n_spinorbitals, n_electrons, mapping, up_then_down):
    """
    Function to initialize the variational parameters of a QMF wave function from a Hartree-Fock
    occupation vector. The theta Bloch angles are derived from an occupation vector, 
    while the phi Bloch angles are randomly selected over the range [0, 2.*np.pi).

    Args:
        n_spinorbitals (int): number of spinorbitals of the molecular system 
        n_electrons (int): number of electrons in system
        mapping (string): specify mapping, see mapping_transform.py for options
            'JW' (Jordan Wigner), or 'BK' (Bravyi Kitaev), or 'SCBK' (symmetry-conserving Bravyi Kitaev)
        up_then_down (boolean): if True, all up, then all down, if False, alternating spin
            up/down
    Returns:
        qmf_var_params (numpy array of complex floats): QMF variational parameter set {Omega}
    """
    vector = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=up_then_down)
    # arrange Bloch angles so all thetas are first then phis
    theta_params = list()
    phi_params = 2. * np.pi * np.random.random((len(vector),)) 
    for occupation in vector:
        theta_params.append(np.pi) if occupation else theta_params.append(0.)
    return np.concatenate((np.array(theta_params), phi_params))


def purify_qmf_state(qmf_var_params):
    """
    The screening procedure for constructing the QCC DIS relies on a z-collinear QMF wave function. 
    If the current QMF wave function is not z-collinear, this function will adjust parameters to the nearest z-collinear
    computational basis state.

    Args:
        qmf_var_params (numpy array of floats): QMF variational parameter set {Omega}

    Returns:
        pure_var_params (numpy array of floats): nearest z-collinear QMF variational parameter set
    """

    print(' Purifying the QMF wave function parameters.\n')
    pure_var_params = qmf_var_params.tolist() 
    n_qmf_params = qmf_var_params.size // 2
    # only need to adjust the theta Bloch angles
    for i, theta in enumerate(qmf_var_params[:n_qmf_params]):
        c_0, c_1 = cos(0.5 * theta), sin(0.5 * theta)
        if abs(c_0) > abs(c_1):
            pure_var_params[i] = 0.
        elif abs(c_0) < abs(c_1):
            pure_var_params[i] = np.pi
        else:
            pure_var_params[i] = choice([0., np.pi])
        print('\tPurified QMF_{:} Bloch angles: (theta, phi) = ({:}, {:}) rad.\n'.format(i, pure_var_params[i], pure_var_params[i + n_qmf_params]))
    return np.array(pure_var_params)

def get_qmf_circuit(qmf_var_params, variational=True):
    """
    Args:
        qmf_var_params (numpy array of floats): QMF variational parameter set {Omega}

    Returns:
        circuit (Circuit): QMF state preparation circuit. 
    """

    circuit = Circuit()
    for index, param in enumerate(qmf_var_params):
        gate_ID = "RX" if index < qmf_var_params.size//2 else "RZ"
        gate = Gate(gate_ID, target=index, parameter=param, is_variational=variational)
        circuit.add_gate(gate)
    return circuit

