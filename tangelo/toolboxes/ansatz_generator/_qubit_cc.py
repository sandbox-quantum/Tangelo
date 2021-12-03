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

"""This module implements functions to create the direct interaction set (DIS)
of generators for the qubit coupled cluster (QCC) ansatz and is based on Ref. 1.
The DIS consists of generator groups that are characterized by the magnitude of
the gradient of the QCC energy functional with respect to a variational parameter
tau, |dEQCC/dtau|. Nonzero values of |dEQCC/dtau| imply that an individual
generator will contribute to variational energy lowering. The number of DIS groups
cannot exceed the number of Hamiltonian terms, N, and each DIS group contains
2^nq - 1 generators, where nq is the number of qubits, all with identical values
of |dEQCC/dtau|. By constructing the DIS, it is possible to identify O(N * (2^nq - 1))
generators that are strong energy-lowering candidates for the QCC ansatz at a
cost of O(N) gradient evaluations. In contrast, a brute-force strategy requires
O(4^nq) gradient evaluations.

Refs:
    1. I. G. Ryabinkin, R. A. Lang, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.
"""

import warnings
from itertools import combinations

from tangelo.toolboxes.operators.operators import QubitOperator
from ._qubit_mf import get_op_expval, purify_qmf_state


def construct_dis(qmf_var_params, qubit_ham, qcc_deriv_thresh, verbose=False):
    """Construct the DIS of QCC generators, which proceeds as follows:
    1. Identify the flip indices of all Hamiltonian terms and group terms by flip indices.
    2. Construct a representative generator using flip indices from each candidate DIS group
       and evaluate dEQCC/dtau for all Hamiltonian terms.
    3. For each candidate DIS group, sum dEQCC/dtau for all Hamiltonian terms.
       If |dEQCC/dtau| >= thresh add the candidate DIS group to the DIS.
    4. For all DIS groups, create a complete set of generators made from Pauli X and and an
       odd number of operators.

    Args:
        qmf_var_params (numpy array of float): The QMF variational parameter set {Omega}.
        qubit_ham (QubitOperator): A qubit Hamiltonian.
        qcc_deriv_thresh (float): Threshold of the value of |dEQCC/dtau| for a generator from
            a candidate DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group
            enters the DIS and its generators can be used in the QCC ansatz.
        verbose (bool): Flag for QCC verbosity.

    Returns:
        list of list: The DIS of QCC generators. Each list in dis contains (1) a complete set
            of generators for a DIS group built from Pauli X and an odd number of Y operators that
            act on qubits indexed by all combinations of the flip indices and (2) the value of
            |dEQCC/dtau|.
    """

    if verbose:
        print("Forming the direct interaction set (DIS) of QCC generators.\n")

    # Use a qubit Hamiltonian and QMF parameter set to construct the DIS
    dis = []
    dis_groups = get_dis_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh, verbose=verbose)
    if dis_groups:
        if verbose:
            print(f"The DIS contains {len(dis_groups)} unique generator group(s).\n")
        for i, dis_group in enumerate(dis_groups):
            dis_group_idxs = [int(idxs) for idxs in dis_group[0].split(" ")]
            dis_group_gens = get_gens_from_idxs(dis_group_idxs)
            dis.append([dis_group_gens, abs(dis_group[1])])
            if verbose:
                print_msg = f"DIS group {i} | group size = {len(dis_group_gens)} | "\
                            f"flip index = {dis_group_idxs} | |dEQCC/dtau| = "\
                            f"{abs(dis_group[1])} a.u.\n"
                print(print_msg)
    else:
        warn_msg = f"DIS = NULL. There are no candidate DIS groups where |dEQCC/dtau| "\
                   f">= {qcc_deriv_thresh} a.u.\n"
        warnings.warn(warn_msg, RuntimeWarning)
    return dis


def get_dis_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh, verbose=False):
    """Purify the QMF variational parameter set and then construct unique DIS groups
    characterized by the flip indices and |dEQCC/dtau|.

    Args:
        qmf_var_params (numpy array of float): The QMF variational parameter set {Omega}.
        qubit_ham (QubitOperator): A qubit Hamiltonian.
        qcc_deriv_thresh (float): The threshold of |dEQCC/dtau| for a generator from a candidate
            DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group enters the
            DIS and its generators can be selected for the QCC ansatz.
        verbose (bool): Flag for QCC verbosity.

    Returns:
        list of tuple: the flip indices (str) and the value of |dEQCC/dtau|
            (float) for DIS groups where |dEQCC/dtau| >= qcc_deriv_thresh.
    """

    # Purify the QMF wave function in order to efficiently screen the DIS
    pure_var_params = purify_qmf_state(qmf_var_params, verbose=verbose)

    # Get the flip indices from qubit_ham and compute the gradient dEQCC/dtau
    qubit_ham_gen = ((term_coef[0], (term_coef[1], pure_var_params))\
       for term_coef in qubit_ham.terms.items() if len(term_coef) > 1)
    flip_idxs = list(filter(None, (get_idxs_deriv(q_gen[0], *q_gen[1])\
        for q_gen in qubit_ham_gen)))

    candidates = {}
    # Group Hamiltonian terms with the same flip indices and sum signed dEQCC/tau values
    for idxs in flip_idxs:
        deriv_old = candidates.get(idxs[0], 0.)
        candidates[idxs[0]] = idxs[1] + deriv_old

    # Return a sorted list of flip indices and |dEQCC/dtau| for each DIS group
    dis_groups = [idxs_deriv for idxs_deriv in candidates.items()\
        if abs(idxs_deriv[1]) >= qcc_deriv_thresh]
    return sorted(dis_groups, key=lambda deriv: abs(deriv[1]), reverse=True)


def get_idxs_deriv(qubit_ham_term, *coef_pure_params):
    """Find the flip indices of a qubit Hamiltonian term by identifying the indices
    of any X and Y operators that are present. A representative generator is then
    built with a Pauli Y operator acting on the first flip index and then Pauli X
    operators acting on the remaining flip indices. Then dEQCC/dtau is evaluated as
    dEQCC_dtau = -i/2 <QMF|[H, gen]|QMF> = -i <QMF|H * gen|QMF>.

    Args:
        qubit_ham_term (tuple of tuple): A QubitOperator term of a Hamiltonian
            specifying its Pauli operators and indices.
        coef_pure_params (tuple): Arguments needed to evaluate the expectation value
            of the QCC energy gradient: coefficient of the term (float) and a purified
            QMF parameter set (numpy array of float).

    Returns:
        tuple or None: if a qubit Hamiltonian term has two or more flip indices, return
            a tuple of the flip indices (str) and the signed value of dEQCC/dtau (float).
            Otherwise return None.
    """

    idxs, gen_list = "", []
    qubit_ham_coef, pure_var_params = coef_pure_params
    for pauli_factor in qubit_ham_term:
       # The index of X or Y operators in the qubit Hamiltonian is a flip index
        idx, pauli_op = pauli_factor
        if "X" in pauli_op or "Y" in pauli_op:
            gen = (idx, "Y") if idxs == "" else (idx, 'X')
            idxs = idxs + f" {idx}" if idxs != "" else f"{idx}"
            gen_list.append(gen)
    if len(idxs.split(" ")) > 1:
        qubit_ham_term = QubitOperator(qubit_ham_term, -1.j * qubit_ham_coef)
        qubit_ham_term *= QubitOperator(tuple(gen_list), 1.)
        deriv = get_op_expval(qubit_ham_term, pure_var_params).real
        idxs_deriv = (idxs, deriv)
    else:
        idxs_deriv = None
    return idxs_deriv


def get_gens_from_idxs(group_idxs):
    """Given the flip indices of a DIS group, create all possible Pauli words made
    from Pauli X and an odd number of Y operators acting on qubits indexed by the
    flip indices.

    Args:
        group_idxs (str): A set of flip indices for a DIS group.

    Returns:
        list of QubitOperator: DIS group generators.
    """

    dis_group_gens, odds = [], list(range(1, len(group_idxs), 2))
    for n_y in odds:
        # Create a list of Y operator indices.
        xy_idxs = list(combinations(group_idxs, n_y))
        for xy_idx in xy_idxs:
            gen_list = []
            for idx in group_idxs:
                # if a flip index idx matches xy_idx, add a Y operator
                gen = (idx, 'Y') if idx in xy_idx else (idx, 'X')
                gen_list.append(gen)
            dis_group_gens.append(QubitOperator(tuple(gen_list), 1.))
    return dis_group_gens
