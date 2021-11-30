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
of generators for the qubit coupled cluster (QCC) ansatz and is based on Ref. 1
below. The DIS consists of generator groups that are characterized by the
magnitude of the QCC energy gradient with respect to a variational paramter tau,
|dEQCC/dtau|. Generators in a DIS group are characterized by nonzero gradients
and can contribute to variational energy lowering. The number of DIS groups
cannot exceed the number of Hamiltonian terms, N, and each DIS group contains
2^nq - 1 generators, where nq is the number of qubits. By constructing the DIS,
it is possible to identify O(N * 2^nq - 1) generators that are strong energy-
lowering candidates for the QCC ansatz at a cost of O(N) gradient evaluations.
In constrast, a brute force strategy requires O(4^nq) gradient evaluations.

Refs:
    1. I. G. Ryabinkin, R. A. Lang, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.
"""

import warnings
from random import choice
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
    4. For all DIS groups in the DIS, create the full set of generators made from Pauli X and Y
       operators and with an odd number of Y operators.

    Args:
        qmf_var_params numpy array of float: The QMF variational parameter set {Omega}.
        qubit_ham (QubitOperator): A qubit Hamiltonian.
        qcc_deriv_thresh (float): Threshold of |dEQCC/dtau| for a generator from a candidate
            DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group enters the
            DIS and its generators can be selected for the QCC ansatz.
        verbose (bool): Flag for QCC verbosity.

    Returns:
        dis list of list: The DIS of QCC generators; each list in dis contains (1) all possible
            generators for a DIS group that are created by permutating Pauli X and Y operators
            on the flip indices and have an odd number of Y operators and (2) the value of
            |dEQCC/dtau|.
    """

    if verbose:
        print("Forming the direct interaction set (DIS) of QCC generators.\n")

    # Use a qubit Hamiltonian and QMF parameter set to construct the DIS
    dis = []
    dis_groups = get_dis_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh, verbose=verbose)
    if dis_groups:
        if verbose:
            print(f"The DIS contains {len(dis_groups)} unique generator group(s)\n")
        for i, dis_group in enumerate(dis_groups):
            group_idxs = choice(list(dis_group[1])) if isinstance(dis_group[1][0], list)\
                else list(dis_group[1])
            dis_group_gens = get_gens_from_idxs(group_idxs)
            dis.append([dis_group_gens, abs(dis_group[0])])
            if verbose:
                print_msg = f"DIS group {i} | group size = {len(dis_group_gens)} | "\
                            f"flip index = {dis_group[1]} | |dEQCC/dtau| = "\
                            f"{abs(dis_group[i])} a.u.\n"
                print(print_msg, RuntimeWarning)
    else:
        warn_msg = f"DIS = NULL. There are no generators where |dEQCC/dtau| \n"\
                   f"> {qcc_deriv_thresh} a.u.\n"
        warnings.warn(warn_msg, RuntimeWarning)
    return dis


def get_dis_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh, verbose=False):
    """Purify the QMF variational parameter set and then construct unique DIS groups
    characterized by the flip indices and |dEQCC/dtau|.

    Args:
        qmf_var_params numpy array of float: The QMF variational parameter set {Omega}.
        qubit_ham (QubitOperator): A qubit Hamiltonian.
        qcc_deriv_thresh (float): The threshold of |dEQCC/dtau| for a generator from a candidate
            DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group enters the
            DIS and its generators can be selected for the QCC ansatz.
        verbose (bool): Flag for QCC verbosity.

    Returns:
        dis_groups list of tuple: A tuple for each DIS group containing |dEQCC/dtau| (float)
            and flip indices as a list of int.
    """

    # Purify the QMF wave function in order to efficiently screen the DIS
    pure_var_params = purify_qmf_state(qmf_var_params, verbose=verbose)
    n_qubits = pure_var_params.size // 2

    # Get the flip indices from qubit_ham and compute the gradient dEQCC/dtau
    qubit_ham_gen = ((term_coef[0], (term_coef[1], pure_var_params, n_qubits))\
        for term_coef in qubit_ham.terms.items())
    flip_lst = list(map(lambda q_gen: get_flip_idx(q_gen[0], *q_gen[1]), qubit_ham_gen))
    flip_data = [flip for flip in flip_lst if flip is not None]

    # Use a dictionary to combine molecular Hamiltonian terms belonging to the same DIS group
    dis_groups = {}
    for data in flip_data:
        idx, deriv, group_idxs = data
        try:
            # If a flip index has previously been seen, update the value of dEQCC/dtau
            dis_groups_old = dis_groups[idx]
            dis_groups[idx] = (deriv + dis_groups_old[0], group_idxs)
        except KeyError:
            dis_groups[idx] = (deriv, group_idxs)

    # Return a sorted list containing |dEQCC/dtau| and the flip index for each DIS group
    dis_groups = [dis for dis in dis_groups.values() if abs(dis[0]) >= qcc_deriv_thresh]
    return sorted(dis_groups, key=lambda x: abs(x[0]), reverse=True)


def get_flip_idx(qubit_ham_term, *flip_idx_data):
    """Finds the flip indices of a qubit Hamiltonian term by iterating through the Pauli
    factors and identifying the indices where X and Y operators appear. A representative
    QCC generator is then built by placing a Y operator at the first flip index and then
    X operators at all remaining flip indices. Then dEQCC/dtau is evaluated as
    dEQCC_dtau = -i/2 <QMF| [H, generator] |QMF> = -i <QMF| H * generator |QMF>.

    Args:
        qubit_ham_term tuple of tuple: A QubitOperator term from the qubit Hamiltonian
            specifying the index and Pauli operator of each term factor.
        flip_idx_data (tuple): qubit_ham_coeff (coefficient of qubit_ham_term), a purified
            QMF paramter set (pure_var_params), and the number of qubits (n_qubits).

    Returns:
        flip_data (tuple): A tuple containing the flip indices (str), dEQCC/dtau (float),
            and a list of ints specifying a DIS group flip index.
    """

    qubit_ham_coef, pure_var_params, n_qubits = flip_idx_data
    idx, group_idxs, gen_list, flip_data = str(), [], [], None
    for i in range(n_qubits):
        # The index of X or Y operators in the qubit Hamiltonian is a flip index
        if (i, "X") in qubit_ham_term or (i, "Y") in qubit_ham_term:
            gen = (i, "Y") if idx == "" else (i, 'X')
            gen_list.append(gen)
            idx += str(i) + str(" ")
            group_idxs.append(i)

    if len(group_idxs) > 1:
        qubit_ham_term = QubitOperator(qubit_ham_term, -1.j * qubit_ham_coef)
        qubit_ham_term *= QubitOperator(tuple(gen_list), 1.)
        deriv = get_op_expval(qubit_ham_term, pure_var_params).real
        flip_data = (idx, deriv, group_idxs)
    return flip_data


def get_gens_from_idxs(group_idxs):
    """Given the flip indices for a DIS group, create all possible QCC generators made of
    Pauli X and Y operators at each flip index.

    Args:
        group_idxs list of int: the DIS group flip indices

    Returns:
        dis_group_gens list of QubitOperator: a list of generators for the DIS group flip indices
    """

    # Note: there must be an odd number of Y operators in each QCC generator
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

