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
from tangelo.linq import Circuit, Gate

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
    """Generate the mapping from Majorana modes to qubit operators

    The algorithm is described in arXiv:1910.10746v2

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
    full_map = dict()
    for i, v in enumerate(all_list[:-1]):
        full_map[i] = QubitOperator(v)
    return full_map


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


def jkmn_vaccuum_circuit(n_qubits, jkmn_map=None):
    """Generate the circuit that initializes vaccuum state for JKMN mapping

    Args:
        n_qubits (int) : The number of qubits in the system
        jkmn_map (dict) : The mapping from a Majorana operator index to a QubitOperator

    Returns:
        Circuit : The circuit that prepares the vaccuum state.
    """

    if jkmn_map is None:
        jkmn_map = _jkmn_dict(n_qubits)
    fock_dict = dict()
    rot_dict = dict()
    # generate number operators
    for i in range(n_qubits):
        fock_dict[i] = 0.5 * QubitOperator([]) - 0.5j * jkmn_map[2*i] * jkmn_map[2*i+1]
        for k, v in fock_dict[i].terms.items():
            for i in k:
                if i[0] not in rot_dict:
                    rot_dict[i[0]] = i[1]
    gate_list = []
    for k, v in rot_dict.items():
        if v == "X":
            gate_list.append(Gate("H", int(k)))
    return Circuit(gate_list, n_qubits=n_qubits)


def jkmn_prep_circuit(vector):
    r"""Generate the circuit corresponding to a HF state with occupations defined by given occupation vector

    The vaccuum state preparation circuit is obtained. Each fermionic mode i is then generated by applying
    \gamma_{2*i} = a_i^{\dagger} + a_i

    Args:
        vector (list of int) : The occupation of each spinorbital

    Returns:
        Circuit : The state preparation circuit for a HF state with occupation given by vector
    """
    n_qubits = len(vector)
    jkmn_map = _jkmn_dict(n_qubits)
    vacuum_circuit = jkmn_vaccuum_circuit(n_qubits, jkmn_map)

    state_prep_qu_op = QubitOperator([], 1)
    for i, occ in enumerate(vector):
        if occ == 1:
            state_prep_qu_op *= jkmn_map[2*i]

    gate_list = []
    for k, v in state_prep_qu_op.terms.items():
        for i in k:
            gate_list.append(Gate(i[1], i[0]))
    return vacuum_circuit + Circuit(gate_list)
