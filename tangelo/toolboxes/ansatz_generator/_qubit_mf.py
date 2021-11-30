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
ansatz: (1) analytically evaluate an expectation value of a QubitOperator
using a QMF wave function; (2) initialize the QMF variational parameter set
{Omega} from a Hartree-Fock reference state; (3) purify {Omega} when building
and screening the DIS of QCC generators; (4) build a QMF state preparation
circuit using {Omega}; (5) create penalty terms for N, S^2, and Sz to penalize
a mean field Hamiltonian. For more information, see references below.

Refs:
    1. I. G. Ryabinkin and S. N. Genin.
        https://arxiv.org/abs/1812.09812 2018.
    2. S. N. Genin, I. G. Ryabinkin, and A. F. Izmaylov.
          https://arxiv.org/abs/1901.04715 2019.
    3. I. G. Ryabinkin, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2019, 15, 1, 249–255.
"""

from functools import reduce
import numpy as np

from tangelo.backendbuddy import Circuit, Gate
from tangelo.toolboxes.operators.operators import FermionOperator
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_vector
from .penalty_terms import combined_penalty, number_operator_penalty, spin2_operator_penalty,\
    spin_operator_penalty


def get_op_expval(qubit_op, qmf_var_params):
    """Driver funtion for analytical evaluation of a QubitOperator expectation value with a
    QMF wave function.

    Args:
        qubit_op (QubitOperator): A qubit operator to compute the expectation value of.
        qmf_var_params numpy array of float: The QMF variational parameter set {Omega}.

    Returns:
        val (complex float): Sum of expectation values for all terms in qubit_op.
    """

    n_qubits = qmf_var_params.size // 2
    qubit_op_data = ((term_coef[0], (term_coef[1], qmf_var_params, n_qubits))\
        for term_coef in qubit_op.terms.items())
    val = list(map(lambda qop_data: calc_op_expval(qop_data[0], *qop_data[1]), qubit_op_data))
    return reduce(lambda x, y:x+y, val) if val else complex(0.)


def calc_op_expval(qubit_op_term, *qop_qmf_data):
    """Analytically evaluate a qubit operator expectation value with a QMF wave function.
    The expectation values of Pauli operators X, Y, and Z are (Ref. 2)
    <QMF| X | QMF> = cos(phi) * sin(theta)
    <QMF| Y | QMF> = sin(phi) * sin(theta)
    <QMF| Z | QMF> = cos(theta)

    Args:
        qubit_op_term tuple of tuple: A QubitOperator term from a qubit operator specifying
            the index and Pauli operator of each term factor.
        qop_qmf_data (tuple): qubit_op_coeff (coefficient of qubit_op_term), the QMF
            variational paramter set (qmf_var_params), and the number of qubits (n_qubits).

    Returns:
        qubit_op_coef (complex float): The expectation value of qubit_op_term.
    """

    qubit_op_coef, qmf_params, n_qubits = qop_qmf_data
    for idx, pauli in qubit_op_term:
        theta, phi = qmf_params[idx], qmf_params[idx + n_qubits]
        if pauli == "X":
            qubit_op_coef *= np.cos(phi) * np.sin(theta)
        elif pauli == "Y":
            qubit_op_coef *= np.sin(phi) * np.sin(theta)
        elif pauli == "Z":
            qubit_op_coef *= np.cos(theta)
    return qubit_op_coef


def init_qmf_state_from_hf_vec(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=0):
    """Function to initialize the QMF variational parameter set {Omega} from a Hartree-Fock state
    occupation vector. The theta Bloch angles are set to 0. or np.pi if the molecular orbital is
    unoccupied or occupied, respectively. The phi Bloch angles are set to 0.

    Args:
        n_spinorbitals (int): Number of spinorbitals of the molecular system.
        n_electrons (int): Number of electrons in system.
        mapping (str) : One of the supported qubit mapping identifiers.
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down.
        spin (int): 2*S = n_alpha - n_beta.

    Returns:
        qmf_var_params numpy array of float: The QMF variational parameter set {Omega}.
    """

    # get thetas from HF vec and arrange Bloch angles so all thetas are first then phis
    thetas = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=up_then_down, spin=spin)
    return np.concatenate((np.pi * thetas, np.zeros((len(thetas),), dtype=float)))


def purify_qmf_state(qmf_var_params, verbose=False):
    """The efficient construction and screening of the DIS requires a z-collinear QMF state.
    If the QMF state specified by qmf_var_params is not z-collinear, this function adjusts the
    parameters to the nearest z-collinear computational basis state.

    Args:
        qmf_var_params numpy array of float: The QMF variational parameter set {Omega}.
        verbose (bool): Flag for QMF verbosity.

    Returns:
        pure_var_params numpy array of float: A purified QMF parameter set that corresponds
            to the nearest z-collinear state to the current QMF state.
    """

    if verbose:
        print("Purifying the QMF wave function parameters.\n")
    pure_var_params, n_qubits = qmf_var_params.tolist(), qmf_var_params.size // 2
    # only need to adjust the theta Bloch angles
    for i, theta in enumerate(qmf_var_params[:n_qubits]):
        c_0, c_1 = np.cos(0.5 * theta), np.sin(0.5 * theta)
        if abs(c_0) > abs(c_1):
            pure_var_params[i] = 0.
        elif abs(c_0) < abs(c_1):
            pure_var_params[i] = np.pi
        else:
            pure_var_params[i] = 0.5 * np.pi
        if verbose:
            print_msg = f"Purified QMF_{i} Bloch angles: (theta, phi) = ({pure_var_params[i]},"\
                        f" {pure_var_params[i + n_qubits]})\n"
            print(print_msg)
    return np.array(pure_var_params)


def get_qmf_circuit(qmf_var_params, variational=True):
    """Build a QMF state preparation circuit using the current state of the QMF variational
    parameter set {Omega}. The first n_qubit elements in {Omega} are the theta Bloch angles
    that serve as paramters for RX gates. The second n_quit elements in {Omega} are
    the phi Bloch angles that are used as parameters for RZ gates.

    Args:
        qmf_var_params (numpy array of floats): The QMF variational parameter set {Omega}.
        variational (bool): Flag to treat {Omega} variationally or not.

    Returns:
        circuit (Circuit): A QMF state preparation circuit.
    """

    n_qubits, circuit = qmf_var_params.size // 2, Circuit()
    for index, param in enumerate(qmf_var_params):
        gate_id = "RX" if index < n_qubits else "RZ"
        circuit.add_gate(Gate(gate_id, target=index, parameter=param, is_variational=variational))
    return circuit


def penalize_mf_ham(mf_pen_terms, n_orbitals, n_electrons, up_then_down=False):
    """Generate a FermionOperator that is used to penalize a mean field Hamiltonian for at
    least one of the N, S^2, or Sz operators. See penalty_terms.py for more details.

    Args:
        mf_pen_terms (dict): The mean field Hamiltonian is penalized using N, S^2, or Sz operators.
            The keys are (str) "init_params", "N", "S^2", and "Sz". The key "init_params" takes one
            of the values in supported_initial_var_params (see qmf.py). For keys "N", "S^2", and
            "Sz", the values are tuples of the penatly term coefficient and the target value of the
            penalty operator: value=(mu, target). Keys and values are case sensitive and mu > 0.
        n_orbitals (int): Number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        n_electrons (int): Number of electrons.
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down.

    Returns:
        mf_penalty (FermionOperator): The sum of all penalty terms to be added to a mean field
            Hamiltonian.
    """

    # get the penalty terms and check there is at least one
    penalty_terms = list(key for key in mf_pen_terms.keys() if key != "init_params")
    if not penalty_terms:
        raise ValueError("Penalty terms were not specified in the mf_pen_terms dictionary.")
    mf_penalty = FermionOperator.zero()
    if len(penalty_terms) == 3:
        mf_penalty += combined_penalty(n_orbs=n_orbitals,\
            opt_penalty_terms=mf_pen_terms, up_then_down=up_then_down)
    else:
        for penalty_term in penalty_terms:
            mu_coef, pen_target = mf_pen_terms[penalty_term]
            if mu_coef < 0:
                raise ValueError("The penalty term coefficient mu must be positive.")
            if penalty_term == "N":
                mf_penalty += number_operator_penalty(n_orbitals, n_electrons=n_electrons,\
                    mu=mu_coef, up_then_down=up_then_down)
            elif penalty_term == "S^2":
                mf_penalty += spin2_operator_penalty(n_orbitals, s2=pen_target, mu=mu_coef,\
                    up_then_down=up_then_down)
            elif penalty_term == "Sz":
                mf_penalty += spin_operator_penalty(n_orbitals, sz=pen_target, mu=mu_coef,\
                    up_then_down=up_then_down)
    return mf_penalty

