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
    Modules for efficient construction of the direct interaction set (DIS)
    of generators for the qubit coupled cluster (QCC) operator.
"""

import numpy as np
from random import choice
from itertools import combinations

from ._qubit_mf import get_op_expval, purify_qmf_state
from tangelo.toolboxes.operators.operators import QubitOperator


def construct_DIS(qmf_var_params, qubit_ham, qcc_deriv_thresh):
    """
    Construct the direct interaction set of QCC generators.

    Args:
        qmf_var_params (numpy array of floats): QMF parameter set {Omega}. 
        qubit_ham (QubitOperator): The qubitized molecular system Hamiltonian.
        qcc_deriv_thresh (float): Threshold for the gradient dEQCC/dtau used to determine whether a particular direct interaction set (DIS) group
                                  of generators should be considered when forming the QCC operator.

    Returns:
            DIS (list of lists): The direct interaction set of QCC generators. The DIS holds lists for each DIS group. Each DIS
                                 group list contains a list of all possible Pauli words for given its defining flip index and the
                                 magnitude of its characteristic gradient dEQCC/dtau.
    """

    print(' Forming the direct interaction set (DIS) of QCC generators.\n')

    # Use a qubitized molecular Hamiltonian and QMF parameter set to construct the DIS
    DIS_groups = get_DIS_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh)

    DIS = list()
    if len(DIS_groups) > 0:
        print(' The DIS contains {:} unique generator group(s):\n'.format(len(DIS_groups)))
        for i, DIS_group in enumerate(DIS_groups):            
            DIS_idx = choice(list(DIS_group[1])) if isinstance(DIS_group[1][0], list) else list(DIS_group[1]) 
            DIS_list = get_DIS_list_from_idx(DIS_idx)
            DIS.append([DIS_list, abs(DIS_group[0])])
            print('\tDIS group {:} | group size = {:} | flip index =  {:} | |dEdtau| = {:} a.u. \n'.format(i, len(DIS_list), DIS_group[1], abs(DIS_group[0])))
    else:
        print(' DIS = NULL: there are no generators where |dEdtau| > {:} a.u.\n'.format(qcc_deriv_thresh))

    return DIS

def get_DIS_groups(qmf_var_params, qubit_ham, qcc_deriv_thresh):
    """
    Construct the DIS groups characterized by flip indices obtain and energy gradient.

    Args:
        qmf_var_params (numpy array of floats): QMF parameter set {Omega}.
        qubit_ham (QubitOperator): The qubitized molecular system Hamiltonian.
        qcc_deriv_thresh (float): Threshold for the gradient dEQCC/dtau used to determine whether a particular direct interaction set (DIS) group
                                  of generators should be considered when forming the QCC operator.

    Returns:
        DIS_groups (list of tuples): each tuple contains the magnitude of the energy gradient and a list of flip indices.
    """

    # Purify the QMF wave function in order to efficiently screen the DIS.
    pure_var_params = purify_qmf_state(qmf_var_params)

    # Get the flip indices from qubit_ham and compute the gradient dEQCC/dtau
    def op_data():
        for term, coef in qubit_ham.terms.items(): 
            # at least 2 flip indices are required for a DIS group
            if term:
                yield term, coef, pure_var_params
    flip_data = list(data for data in map(get_flip_idx, op_data()) if data is not None)

    # Use a dictionary to combine molecular Hamiltonian terms belonging to the same DIS group.
    DIS_groups = dict()
    for data in flip_data:
        idx, dEQCC_dtau, idx_list = data 
        try:
            # If a flip index has previously been seen, update the value of dEQCC/dtau.
            DIS_groups_old = DIS_groups[idx]
            DIS_groups[idx] = (dEQCC_dtau + DIS_groups_old[0], idx_list)
        except KeyError:
            DIS_groups[idx] = (dEQCC_dtau, idx_list)
    DIS_groups = list(DIS_list for DIS_list in DIS_groups.values() if abs(DIS_list[0]) >= qcc_deriv_thresh)

    # Return a list sorted by |dEQCC/dtau| containing the flip indices and |dEQCC/dtau| for each DIS group
    return sorted(DIS_groups, key=lambda x: abs(x[0]), reverse=True)

def get_flip_idx(args):
    """
    Finds the flip index for a given molecular Hamiltonian term.

    Args:
        args (tuple): a tuple containing a qubit Hamiltonian term, its coefficient,
                      a purified QMF variational parameter set, and the number of qubits.
    Returns:
        flip_data (tuple): a tuple containing the flip index (str), the expectation value of dEQCC/dtau = -0.5j <QMF|[H, gen]|QMF>
                                        where H_j is a molecular Hamiltonian term and P_k is a QCC generator Pauli word, and a list of integers 
                                        specifying the flip index.
    """

    H_term, coef, pure_var_params = args 
    n_qubits, idx, idx_list, generator, flip_data = pure_var_params.size // 2, str(), list(), list(), None
    for i in range(n_qubits):
        # The index of X or Y operators in the qubit Hamiltonian defines the flip index
        if (i, "X") in H_term or (i, "Y") in H_term:
            generator.append((i, "Y")) if idx == "" else generator.append((i, 'X'))
            idx += str(i) + str(" ")
            idx_list.append(i)

    if len(idx_list) > 1:
        # Evaluate dEQCC/dtau = -0.5j <QMF|[H, gen]|QMF> = <QMF|H * gen|QMF> or 0.
        dEQCC_dtau = get_op_expval(-1.j * QubitOperator(H_term, coef) * QubitOperator(tuple(generator), 1.), pure_var_params).real
        flip_data = (idx, dEQCC_dtau, idx_list)
    
    return flip_data 

def get_DIS_list_from_idx(idx_list):
    """
    Output a list of QCC generators for a given flip index.

    Args:
        idx_list (list of ints): a list containing the flip indices

    Returns:
        DIS_list (list of QubitOperators): a list of QCC generators for a given flip index 
    """

    DIS_list = list()
    # Create a list containing odd numbers bounded by the number of flip indices
    # Note: there must be an odd number of Y operators in each QCC generator
    odds = list(i+1 for i in range(0, len(idx_list), 2))
    for Ny in odds:
        # Create a list of Y operator indices.
        XY_idx = list(combinations(idx_list, Ny))
        for xy_idx in XY_idx:
            generator = list() 
            for idx in idx_list:
                # if a flip index idx matches xy_idx, add a Y operator
                generator.append((idx, 'Y')) if (idx in xy_idx) else generator.append((idx, 'X'))
            DIS_list.append(QubitOperator(tuple(generator), 1.))
    return DIS_list


