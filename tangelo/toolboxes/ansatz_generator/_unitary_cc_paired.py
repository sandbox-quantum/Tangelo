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

"""This module defines the fermionic operators that can be used to generate
the UpCCGSD ansatz.
"""

from tangelo.toolboxes.ansatz_generator._general_unitary_cc import get_spin_ordered
from tangelo.toolboxes.operators import FermionOperator


def get_upccgsd(n_orbs, values, up_then_down=False, anti_hermitian=True):
    r"""Function to generate one UpCCGSD layer as a FermionicOperator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        values (list): parameters for 1-UpCCGSD
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True) openfermion (False)
        anit_hermitian (bool): Whether to include the anti_hermitian conjugate in the operator

    Returns:
        ferm_op (FermionicOperator): The 1-UpCCGSD ansatz as a fermionic operator"""

    all_terms = get_generalized_singles(n_orbs, up_then_down) + get_paired_doubles(n_orbs, up_then_down)

    ferm_op = FermionOperator()
    for i, item in enumerate(all_terms):
        ferm_op += FermionOperator(item[0], values[i])
        if anti_hermitian:
            if len(item[0]) == 2:
                ferm_op += FermionOperator(((item[0][1][0], 1), (item[0][0][0], 0)), -values[i])
            else:   # length is 4, i.e. generalized paired double term
                ferm_op += FermionOperator(((item[0][3][0], 1), (item[0][2][0], 0),
                                            (item[0][1][0], 1), (item[0][0][0], 0)), -values[i])
    return ferm_op


def get_generalized_singles(n_orbs, up_then_down=False):
    r"""Function to obtain the generalized singles fermionic generator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True) openfermion (False)

    Returns:
        all_terms (list): The generalized singles term \sum_{ij} t_{ij} a_i^{\dagger} a_j"""

    all_terms = list()
    for i in range(n_orbs):
        for j in range(i+1, n_orbs):
            up, dn = get_spin_ordered(n_orbs, j, i, up_down=up_then_down)  # get spin-orbital indices
            all_terms.extend([[((up[0], 1), (up[1], 0)), 1], [((dn[0], 1), (dn[1], 0)), 1]])
    return all_terms


def get_paired_doubles(n_orbs, up_then_down=False):
    r"""Function to obtain the generalized paired doubles fermionic generator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        up_then_down (bool): The ordering of the spin orbitals. qiskit (True) openfermion (False)

    Returns:
        all_terms (list): The paired doubles term \sum_{ij} t_{ij} a_i^+ a_j a_i^+ a_j"""

    all_terms = list()
    for i in range(n_orbs):
        for j in range(i+1, n_orbs):
            up, dn = get_spin_ordered(n_orbs, j, i, up_down=up_then_down)  # get spin-orbital indices
            all_terms.extend([[((up[0], 1), (up[1], 0), (dn[0], 1), (dn[1], 0)), 1]])
    return all_terms
