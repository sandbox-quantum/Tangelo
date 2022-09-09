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

"""This module implements a collection of functions related to the QMF
ansatz:
    1. Analytical evaluation of an expectation value of a QubitOperator
       using a QMF wave function;
    2. Initialization of the QMF variational parameter set {Omega} from a
       Hartree-Fock reference state;
    3. Purification {Omega} when building and screening the DIS of QCC generators;
    4. Construction of a QMF state circuit using {Omega};
    5. Addition of terms for N, S^2, and Sz that penalize a mean-field Hamiltonian
       in order to obtain solutions corresponding to specific electron number
       and spin symmetries.
For more information, see references below.

Refs:
    1. I. G. Ryabinkin and S. N. Genin.
        https://arxiv.org/abs/1812.09812 2018.
    2. S. N. Genin, I. G. Ryabinkin, and A. F. Izmaylov.
          https://arxiv.org/abs/1901.04715 2019.
    3. I. G. Ryabinkin, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2019, 15, 1, 249–255.
"""

import numpy as np

from tangelo.linq import Circuit, Gate
from tangelo.toolboxes.operators.operators import FermionOperator
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector, get_mapped_vector
from tangelo.toolboxes.ansatz_generator.penalty_terms import combined_penalty, number_operator_penalty,\
                                                             spin2_operator_penalty, spin_operator_penalty


def get_op_expval(qubit_op, qmf_var_params):
    """Driver function for analytical evaluation of a QubitOperator expectation value with a
    QMF wave function.

    Args:
        qubit_op (QubitOperator): A qubit operator to compute the expectation value of.
        qmf_var_params (numpy array of float): QMF variational parameter set.

    Returns:
        complex: expectation value of all qubit operator terms.
    """

    n_qubits = qmf_var_params.size // 2
    qubit_op_gen = ((qop_items[0], (qop_items[1], qmf_var_params, n_qubits))
                     for qop_items in qubit_op.terms.items())
    return sum([calc_op_expval(qop_gen[0], *qop_gen[1]) for qop_gen in qubit_op_gen])


def calc_op_expval(qop_term, *qop_qmf_data):
    """Analytically evaluate a qubit operator expectation value with a QMF wave function.
    The expectation values of Pauli operators X, Y, and Z are (Ref. 2)
    <QMF| X | QMF> = cos(phi) * sin(theta), <QMF| Y | QMF> = sin(phi) * sin(theta),
    <QMF| Z | QMF> = cos(theta)

    Args:
        qop_term (tuple of tuple): The Pauli operators and indices of a QubitOperator term.
        qop_qmf_data (tuple): The coefficient of a QubitOperator term, QMF variational parameter
            set (numpy array of float), and number of qubits (int).

    Returns:
        complex: expectation value of a qubit operator term.
    """

    coef, qmf_var_params, n_qubits = qop_qmf_data
    for idx, pauli in qop_term:
        theta, phi = qmf_var_params[idx], qmf_var_params[idx + n_qubits]
        if pauli == "X":
            coef *= np.cos(phi) * np.sin(theta)
        elif pauli == "Y":
            coef *= np.sin(phi) * np.sin(theta)
        elif pauli == "Z":
            coef *= np.cos(theta)
    return coef


def init_qmf_from_hf(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    """Function to initialize the QMF variational parameter set from a Hartree-Fock state
    occupation vector. The theta Bloch angles are set to 0. or np.pi if the molecular orbital is
    unoccupied or occupied, respectively. The phi Bloch angles are set to 0.

    Args:
        n_spinorbitals (int): Number of spin-orbitals in the molecular system.
        n_electrons (int): Number of electrons in the molecular system.
        mapping (str) : One of the supported qubit mapping identifiers.
        up_then_down (bool): Change basis ordering putting all spin-up orbitals first,
            followed by all spin-down.
        spin (int): 2*S = n_alpha - n_beta.

    Returns:
        numpy array of float: QMF variational parameter set.
    """

    # Get thetas from HF vec and arrange Bloch angles so all thetas are first then phis
    thetas = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down, spin)
    return np.concatenate((np.pi * thetas, np.zeros((len(thetas),), dtype=float)))


def init_qmf_from_vector(vector, mapping, up_then_down=False):
    """Function to initialize the QMF variational parameter set from a Hartree-Fock state
    occupation vector. The theta Bloch angles are set to 0. or np.pi if the molecular orbital is
    unoccupied or occupied, respectively. The phi Bloch angles are set to 0.

    Args:
        vector (array): Occupation vector of orbitals using alternating up and down electrons (i.e. up_then_down=False)
        mapping (str): One of the supported qubit mapping identifiers.
        up_then_down (bool): Change basis ordering putting all spin-up orbitals first,
            followed by all spin-down when applying the qubit mapping.

    Returns:
        numpy array of float: QMF variational parameter set.
    """

    # Get thetas from HF vec and arrange Bloch angles so all thetas are first then phis
    thetas = np.array(get_mapped_vector(vector, mapping, up_then_down))
    var_params = np.zeros(2*len(thetas))
    var_params[:len(thetas)] = thetas
    return var_params


def purify_qmf_state(qmf_var_params, n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    """The efficient construction and screening of the DIS requires a z-collinear QMF state.
    If the QMF state specified by qmf_var_params is not z-collinear, this function adjusts the
    parameters to the nearest z-collinear computational basis state.

    Args:
        qmf_var_params (numpy array of float): QMF variational parameter set.
        n_spinorbitals (int): Number of spin-orbitals in the molecular system.
        n_electrons (int): Number of electrons in the molecular system.
        mapping (str) : One of the supported qubit mapping identifiers.
        up_then_down (bool): Change basis ordering putting all spin-up orbitals first,
            followed by all spin-down.
        spin (int): 2*S = n_alpha - n_beta.

    Returns:
        numpy array of float: purified QMF parameter set that corresponds to the
            nearest z-collinear state to the current QMF state.
    """

    # Adjust the theta Bloch angles
    pure_var_params, n_qubits = np.copy(qmf_var_params), qmf_var_params.size // 2
    for i, theta in enumerate(qmf_var_params[:n_qubits]):
        c_0, c_1 = np.cos(0.5 * theta), np.sin(0.5 * theta)
        if abs(c_0) > abs(c_1):
            pure_var_params[i] = 0.
        elif abs(c_0) < abs(c_1):
            pure_var_params[i] = np.pi
        else:
            vector = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down, spin)
            pure_var_params[i] = np.pi * vector[i]
    return pure_var_params


def get_qmf_circuit(qmf_var_params, variational=True):
    """Build a QMF state preparation circuit using the current state of the QMF variational
    parameter set {Omega}. The first n_qubit elements in {Omega} are parameters for RX gates,
    and the second n_qubit elements in {Omega} are parameters for RZ gates.

    Args:
        qmf_var_params (numpy array of float): QMF variational parameter set.
        variational (bool): Flag to treat {Omega} variationally or keep them fixed.

    Returns:
        Circuit: instance of tangelo.linq Circuit class.
    """

    n_qubits, gates = qmf_var_params.size // 2, []
    for idx, param in enumerate(qmf_var_params):
        gate_id = "RX" if idx < n_qubits else "RZ"
        gates.append(Gate(gate_id, target=idx % n_qubits, parameter=param, is_variational=variational))
    return Circuit(gates)


def penalize_mf_ham(mf_pen_terms, n_orbitals):
    """Generate a FermionOperator that is used to penalize a mean-field Hamiltonian for at
    least one of the N, S^2, or Sz operators.

    Args:
        mf_pen_terms (dict): Parameters for mean-field Hamiltonian penalization.
            The keys are "N", "S^2", or "Sz" (str) and the values are a tuple of the penalty
            term coefficient, mu (float), and the target value of a penalty operator (int).
            Example - "key": (mu, target). Key, value pairs are case sensitive and mu > 0.
        n_orbitals (int): Number of orbitals in the fermion basis.

    Returns:
        FermionOperator: sum of all mean-field Hamiltonian penalty terms.
    """

    mf_penalty = FermionOperator.zero()
    penalty_terms = list(key for key in mf_pen_terms.keys())
    if len(penalty_terms) == 3:
        mf_penalty += combined_penalty(n_orbitals, mf_pen_terms, False)
    else:
        for penalty_term in penalty_terms:
            coef, target = mf_pen_terms[penalty_term]
            if coef <= 0.:
                raise ValueError("The penalty term coefficient must be positive.")
            if penalty_term == "N":
                mf_penalty += number_operator_penalty(n_orbitals, target, coef, False)
            elif penalty_term == "S^2":
                mf_penalty += spin2_operator_penalty(n_orbitals, target, coef, False)
            elif penalty_term == "Sz":
                mf_penalty += spin_operator_penalty(n_orbitals, target, coef, False)
    return mf_penalty
