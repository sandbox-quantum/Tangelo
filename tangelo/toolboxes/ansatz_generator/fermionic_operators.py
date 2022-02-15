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

r"""This module defines the fermionic operators that can be used to obtain
expectation values of commonly used quantum numbers. The available operators are
1) N: number of electrons
2) Sz: The spin z-projection Sz|\psi>=m_s|\psi>
3) S^2: The spin quantum number S^2|\psi>=s(s+1)|\psi> associated with spin
angular momentum which allows one to decide whether the state has the correct
properties.
"""

from tangelo.toolboxes.ansatz_generator._general_unitary_cc import get_spin_ordered
from tangelo.toolboxes.operators import normal_ordered, list_to_fermionoperator


def number_operator(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered number operator as a
    FermionOperator.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down: The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        FermionOperator: The number operator penalty \hat{N}.
    """

    all_terms = number_operator_list(n_orbs, up_then_down)
    num_op = list_to_fermionoperator(all_terms)

    return normal_ordered(num_op)


def number_operator_list(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered number operator as a list.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        list: The number operator penalty \hat{N}.
    """

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.extend([[((up[0], 1), (up[1], 0)), 1], [((dn[0], 1), (dn[1], 0)), 1]])
    return all_terms


def spinz_operator(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered Sz operator.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        FermionOperator: The Sz operator \hat{Sz}.
    """

    all_terms = spinz_operator_list(n_orbs, up_then_down)
    spinz_op = list_to_fermionoperator(all_terms)

    return normal_ordered(spinz_op)


def spinz_operator_list(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered Sz operator as a list.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        list: The Sz operator \hat{Sz}.
    """

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.extend([[((up[0], 1), (up[1], 0)), 1/2], [((dn[0], 1), (dn[1], 0)), -1/2]])
    return all_terms


def spin2_operator(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered S^2 operator as a
    FermionOperator.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        FermionOperator: The S^2 operator \hat{S}^2.
    """

    all_terms = spin2_operator_list(n_orbs, up_then_down)
    spin2_op = list_to_fermionoperator(all_terms)

    return normal_ordered(spin2_op)


def spin2_operator_list(n_orbs, up_then_down=False):
    r"""Function to generate the normal ordered S^2 operator as a list.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True)
            openfermion (False).

    Returns:
        list: The S^2 operator \hat{S}^2.
    """

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.extend([[((up[0], 1), (up[1], 0), (up[0], 1), (up[1], 0)), 1/4],
                         [((dn[0], 1), (dn[1], 0), (dn[0], 1), (dn[1], 0)), 1/4],
                         [((up[0], 1), (up[1], 0), (dn[0], 1), (dn[1], 0)), -1/4],
                         [((dn[0], 1), (dn[1], 0), (up[0], 1), (up[1], 0)), -1/4],
                         [((up[0], 1), (dn[1], 0), (dn[0], 1), (up[1], 0)), 1/2],
                         [((dn[0], 1), (up[1], 0), (up[0], 1), (dn[1], 0)), 1/2]])
        for j in range(n_orbs):
            if (i != j):
                up2, dn2 = get_spin_ordered(n_orbs, j, j, up_down=up_then_down)
                all_terms.extend([[((up[0], 1), (up[1], 0), (up2[0], 1), (up2[1], 0)), 1/4],
                                 [((dn[0], 1), (dn[1], 0), (dn2[0], 1), (dn2[1], 0)), 1/4],
                                 [((up[0], 1), (up[1], 0), (dn2[0], 1), (dn2[1], 0)), -1/4],
                                 [((dn[0], 1), (dn[1], 0), (up2[0], 1), (up2[1], 0)), -1/4],
                                 [((up[0], 1), (dn[1], 0), (dn2[0], 1), (up2[1], 0)), 1/2],
                                 [((dn[0], 1), (up[1], 0), (up2[0], 1), (dn2[1], 0)), 1/2]])
    return all_terms
