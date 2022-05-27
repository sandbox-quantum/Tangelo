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

import numpy as np
from openfermion.transforms.opconversions.conversions import get_majorana_operator

from tangelo.toolboxes.operators import QubitOperator

sigma_map = {"0": "X", "1": "Y", "2": "Z"}


def _node_value(p, l):
    """Obtain the node value from vector p and depth l as given in Eq (3) in
    arXiv:1910.10746v2"""
    nv = (3**l - 1)//2
    for j in range(l):
        nv += 3**(l-1-j)*int(p[j])
    return nv


def _jkmn_list(h):
    """Generate the full list of paths in the JKMN defined ternary tree of height h

    The algorithm is described in arXiv:1910.10746v2

    Args:
        h (int) : The height of the full ternary tree to generate the leaf paths

    Returns:
        list : The list with elements that generate the qubit operators
        for each path in the ternary tree
    """
    t_list = list()
    for i in range(3**h):
        # string representation of leaf index with base 3 and padding 0s to the tree height h
        terstring = np.base_repr(i, base=3).rjust(h, '0')

        # descend ternary tree and obtain tuples describing QubitOperator
        c_qu_op = []
        for ch in range(h):
            # generates tuple that initiates QubitOperator on index nv and operator op=X,Y,Z
            nv = _node_value(terstring, ch)
            op = sigma_map[terstring[ch]]
            c_qu_op += [(nv, op)]
        t_list += [c_qu_op]
    return t_list


def _jkmn_dict(n_qubits):
    r"""Generate the mapping from Majorana modes to qubit operators

    The algorithm is described in arXiv:1910.10746v2

    Although all assignments of leaves to Majorana modes are valid (with one arbitrary leaf being dropped),
    this function assigns leaves to majorana modes via the following process
    1) The furthest right leaf is the mode dropped.
    2) One can obtain the vaccuum state by applying Hadamard transforms to certain qubits.
       This essentially reorders the assignment of leafs to Majorana modes as HXH=Z, HZH=X, HYH=-Y, HH=I
    3) The order is finalized by assigning \gamma_{2*i} with 'X' on qubit i and \gamma_{2*i+1}
       has 'Y' on qubit i. If \gamma_{2*i} and \gamma_{2*i+1} were swapped, a negative sign on one of the
       \gamma_{2*i} or \gamma_{2*i+1} Majorana modes would be required to obtain a valid vaccuum state.

    This process ensures that all number operators a_i^{\dagger}a_i = 1/2*(1 + 1j*\gamma_{2*i}\gamma_{2*j+1})
    result in qubit operators with only Z operators including Z on qubit i.

    Args:
        n_qubits (int) : The number of qubits in the system

    Returns:
        dict : The mapping from Majorana modes to qubit operators
    """

    # Calculate initial height of ternary tree (largest full tree with n < n_qubit nodes)
    h = int(np.log10(2 * n_qubits + 1) / np.log10(3))

    # create full tree of largest size with less than n nodes
    all_list = _jkmn_list(h)

    # Obtain number of leaves of tree that need to be branched
    n_leaves_to_qubit = n_qubits - (3**h - 1) // 2

    # change first n_leaves_to_qubit leaves to qubits
    prepend_list = []
    for j in range(n_leaves_to_qubit):
        # remove leaf and change to node append sigma_x sigma_y sigma_z
        item_to_qubit = all_list.pop(0)
        for i in range(3):
            nv = (3**(h)-1)//2 + j
            prepend_list.append(item_to_qubit+[(nv, sigma_map[str(i)])])

    all_list = prepend_list + all_list

    # Create map from Majorana modes to QubitOperators ignoring furthest right leaf
    jkmn_map = dict()
    for i, v in enumerate(all_list[:-1]):
        jkmn_map[i] = QubitOperator(v)

    # Find indices to apply H transformation HXH=Z, HZH=X, HYH=-Y
    hinds = _jkmn_vaccuum_indices(n_qubits, jkmn_map)

    # Obtain transformed jkmn map and determine \gamma_{2*i} \gamma_{2*i+1} majorana modes to swap
    tjkmn_map = dict()
    for i in range(2*n_qubits):
        t = tuple()
        for term in jkmn_map[i].terms.keys():
            for iterm in term:
                if iterm[0] in hinds:
                    if iterm[1] == 'X':
                        t += ((iterm[0], 'Z'),)
                    elif iterm[1] == 'Z':
                        t += ((iterm[0], 'X'),)
                    else:
                        t += ((iterm[0], iterm[1]), )
                else:
                    t += ((iterm[0], iterm[1]),)
        tjkmn_map[i] = QubitOperator(t)

    # Swap order of majorana modes so that \gamma_{2*i} has 'X' on qubit i and \gamma_{2*i+1} has 'Y' on qubit i
    stjkmn_map = dict()
    for i in range(n_qubits):
        q1 = next(iter(tjkmn_map[2*i].terms))
        q2 = next(iter(tjkmn_map[2*i+1].terms))
        for tup in q1:
            if tup[1] == 'X' and (tup[0], 'Y') in q2:
                stjkmn_map[2*tup[0]] = tjkmn_map[2*i]
                stjkmn_map[2*tup[0]+1] = tjkmn_map[2*i+1]
            elif tup[1] == 'Y' and (tup[0], 'X') in q2:
                stjkmn_map[2*tup[0]+1] = tjkmn_map[2*i]
                stjkmn_map[2*tup[0]] = tjkmn_map[2*i+1]

    return stjkmn_map


def jkmn(fermion_operator, n_qubits):
    """The JKMN mapping of a fermion operator as described in arXiv:1910.10746v2

    Args:
        fermion_operator (FermionOperator) : The fermion operator to transform to a qubit operator
        n_qubits (int) : The number of qubits in the system

    Returns:
        QubitOperator : The qubit operator corresponding to the Fermion Operator
    """
    mj_op = get_majorana_operator(fermion_operator)
    jkmn_map = _jkmn_dict(n_qubits)

    full_operator = QubitOperator()
    for term, coeff in mj_op.terms.items():
        if abs(coeff) > 1.e-12:
            c_term = QubitOperator([], coeff)
            for mj in term:
                c_term *= jkmn_map[mj]
            full_operator += c_term
    full_operator.compress()
    return full_operator


def _jkmn_vaccuum_indices(n_qubits, jkmn_map=None):
    """Generate the indices for which an 'H' transformation is necessary

    Args:
        n_qubits (int) : The number of qubits in the system
        jkmn_map (dict) : The mapping from a Majorana operator index to a QubitOperator

    Returns:
        list : Indices that require an 'H' gate transformation
    """

    if jkmn_map is None:
        jkmn_map = _jkmn_dict(n_qubits)
    fock_dict = dict()
    rot_dict = dict()
    # generate number operators
    for i in range(n_qubits):
        fock_dict[i] = 0.5 * QubitOperator([]) - 0.5j * jkmn_map[2*i] * jkmn_map[2*i+1]
        for k in fock_dict[i].terms:
            for i in k:
                if i[0] not in rot_dict:
                    rot_dict[i[0]] = i[1]

    return [int(k) for k, v in rot_dict.items() if v == "X"]


def jkmn_prep_vector(vector):
    r"""Apply JKMN mapping to fermion occupation vector.

    Each fermionic mode i is generated by applying \gamma_{2*i} = a_i^{\dagger} + a_i. The returned
    vector is defined by which qubits have X or Y operations applied.

    Args:
        vector (list of int) : The occupation of each spinorbital

    Returns:
        list[int] : The state preparation vector that defines which qubits to apply X gates to
    """
    n_qubits = len(vector)
    jkmn_map = _jkmn_dict(n_qubits)

    state_prep_qu_op = QubitOperator([], 1)
    for i, occ in enumerate(vector):
        if occ == 1:
            state_prep_qu_op *= jkmn_map[2*i]

    x_vector = np.zeros(n_qubits, dtype=int)
    for k in state_prep_qu_op.terms:
        for i in k:
            x_vector[i[0]] = 1 if i[1] in ["X", "Y"] else 0
    return list(x_vector)
