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


def _number_to_base(n, b, h):
    if n == 0:
        return "0"*h
    digits = ""
    while n:
        digits += str(n % b)
        n //= b
    while len(digits) < h:
        digits += "0"
    return digits[::-1]


def _node_value(terstring, l):
    if l == 0:
        return 0
    nv = (3**l - 1)//2
    for j in range(l):
        nv += 3**(l-1-j)*int(terstring[j])
    return nv


def _jkmn_list(h):
    t_list = []
    for i in range(3**h):
        terstring = _number_to_base(i, 3, h)
        c_qu_op = []
        for ch in range(h):
            nv = _node_value(terstring, ch)
            c_qu_op += [(nv, sigma_map[terstring[ch]])]
        t_list += [c_qu_op]
    return t_list


def _jkmn_dict(n_qubits):
    n = n_qubits
    h = int(np.log10(2*n + 1)/np.log10(3))
    n_leaves_to_qubit = n - (3**h - 1) // 2

    # create full tree of largest size with less than n nodes
    all_list = _jkmn_list(h)

    # change first n_leaves_to_qubit leaves to qubits
    prepend_list = []
    for j in range(n_leaves_to_qubit):
        # remove leaf and change to node
        item_to_qubit = all_list.pop(0)
        for i in range(3):
            nv = (3**(h)-1)//2 + j
            prepend_list.append(item_to_qubit+[(nv, sigma_map[str(i)])])

    all_list = prepend_list + all_list

    # Create map from Majorana modes to QubitOperators
    full_map = dict()
    for i, v in enumerate(all_list[:-1]):
        # print(f"{i, v}")
        full_map[i] = QubitOperator(v)
    return full_map


def jkmn(fermion_operator, n_qubits):
    mj_op = get_majorana_operator(fermion_operator)
    jkmn_map = _jkmn_dict(n_qubits)

    full_operator = QubitOperator()
    for term, coeff in mj_op.terms.items():
        if abs(coeff) > 1.e-12:
            c_term = QubitOperator([], coeff)
            for mj in term:
                c_term *= jkmn_map[mj]
            full_operator += c_term
    return full_operator


def _jkmn_vaccuum_circuit(full_map, n):
    fock_dict = dict()
    rot_dict = dict()
    # generate number operators
    for i in range(n):
        fock_dict[i] = 0.5 * QubitOperator([]) - 0.5j * full_map[2*i] * full_map[2*i+1]
        for k, v in fock_dict[i].terms.items():
            if k:
                for i in k:
                    if i[0] not in rot_dict:
                        rot_dict[i[0]] = i[1]
    gate_list = []
    for k, v in rot_dict.items():
        if v == "X":
            gate_list.append(Gate("H", int(k)))
    return Circuit(gate_list, n_qubits=n)


def jkmn_prep_circuit(vector, n_qubits):
    jkmn_map = _jkmn_dict(n_qubits)
    vacuum_circuit = _jkmn_vaccuum_circuit(jkmn_map, n_qubits)

    state_prep_qu_op = QubitOperator([], 1)
    for i, occ in enumerate(vector):
        if occ == 1:
            state_prep_qu_op *= jkmn_map[2*i]

    gate_list = []
    for k, v in state_prep_qu_op.terms.items():
        for i in k:
            gate_list.append(Gate(i[1], i[0]))
    return vacuum_circuit + Circuit(gate_list)
