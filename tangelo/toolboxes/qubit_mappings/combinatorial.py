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

"""TODO
"""

import itertools
from collections import OrderedDict
from math import ceil

from openfermion import chemist_ordered
from openfermion.linalg import qubit_operator_sparse, get_sparse_operator
import numpy as np
from scipy.special import comb

from tangelo.linq.helpers.circuits.measurement_basis import pauli_string_to_of
from tangelo.toolboxes.operators import QubitOperator, FermionOperator


def f(sigma, M):
    """TODO

    Args:

    Returns:

    """
    N = len(sigma)

    terms_k = [comb(M-sigma[N-1-k]-1, k+1) for k in range(N)]
    unique_int = comb(M, N) - 1 - np.sum(terms_k)

    if not unique_int.is_integer():
        raise ValueError
    return int(unique_int)

def basis(M, N):
    """TODO

    Args:

    Returns:

    """
    mapping = [(sigma, f(sigma, M)) for sigma in itertools.combinations(range(M), N)]
    return OrderedDict(mapping)

def get_bitstring(sigma, M):
    """TODO

    Args:

    Returns:

    """

    bitstring = np.zeros(M)
    np.put(bitstring, ind=sigma, v=1)

    return bitstring

def get_sigmam(bistring):
    """TODO

    Args:

    Returns:

    """

    sigma = tuple(np.where(bistring == 1)[0])
    M = len(bistring)

    return sigma, M

def op_on_sigma(ops, sigma):
    """ Eq. (20) without the (-1)^p term."""

    assert len(ops) == 2, f"{ops}"

    sigma = list(sigma)

    for i_qubit, creation_op in reversed(ops):
        # If it is a^{\dagger} (creation operator)
        if creation_op:
            if i_qubit not in sigma:

                sigma = [*sigma, i_qubit]
            else:
                return 0
        else:
            if i_qubit in sigma:
                sigma.remove(i_qubit)
            else:
                return 0

    return tuple(sorted(sigma))


def compact_hamiltonian(H_ferm, n_modes, n_electrons, h1, h2):
    """TODO
    up_then_down must be set to False for now.

    Args:

    Returns:

    """

    #assert H_ferm.is_normal_ordered()
    #H_ferm = chemist_ordered(H_ferm)
    #print(H_ferm.constant)

    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError

    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n = ceil(np.log2(n_choose_alpha * comb(n_modes, n_beta, exact=True)))

    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha*n_choose_alpha)+int_beta
            basis_set[sigma] = unique_int

    #print(basis_set)

    # H_1 and H_2 initialization to 2^n * 2^n matrices.
    h_one = np.zeros((2**n, 2**n))
    h_two = np.zeros((2**n, 2**n))

    """
    for op, _ in H_ferm.terms.items():
        for sigma_pp, unique_int in basis_set.items():

            # 1-body terms.
            if len(op) == 2:
                i, j = op[0][0], op[1][0]

                sigma_qq = op_on_sigma(op, sigma_pp)

                if sigma_qq in basis_set.keys():
                    int_p = basis_set[sigma_pp]
                    int_q = basis_set[sigma_qq]

                    h_one[int_p, int_q] += h1[i][j]
                    h_one[int_q, int_p] += h1[j][i].conj()

            # 2-body terms.
            elif len(op) == 4:
                i, j, k, l = op[0][0], op[1][0], op[2][0], op[3][0]

                sigma_qq = op_on_sigma(op[:2], sigma_pp)

                if sigma_qq in basis_set.keys():
                    sigma_tt = op_on_sigma(op[2:], sigma_qq)

                    if sigma_tt in basis_set.keys():
                        int_p = basis_set[sigma_pp]
                        int_t = basis_set[sigma_tt]

                        h_two[int_p, int_t] += h2[i][j][k][l]
                        h_two[int_t, int_p] += h2[k][l][i][j].conj()

                j, k = op[1][0], op[2][0]
                if j == k:
                    raise ValueError
    """

    for sigma_pp, unique_int in basis_set.items():

        # 1-body terms.
        for i, j in itertools.product(range(2*n_modes), repeat=2):
            op = ((i, 1), (j, 0))
            sigma_qq = op_on_sigma(op, sigma_pp)

            if sigma_qq in basis_set.keys():
                int_p = basis_set[sigma_pp]
                int_q = basis_set[sigma_qq]

                h_one[int_p, int_q] += h1[i][j]
                h_one[int_q, int_p] += h1[j][i].conj()

    #for sigma_pp, unique_int in basis_set.items():
        # 2-body terms.
        for i, j, k, l in itertools.product(range(2*n_modes), repeat=4):
            op = ((i, 1), (j, 0), (k, 1), (l, 0))
            sigma_qq = op_on_sigma(op[:2], sigma_pp)

            if sigma_qq in basis_set.keys():
                sigma_tt = op_on_sigma(op[2:], sigma_qq)

                if sigma_tt in basis_set.keys():
                    int_p = basis_set[sigma_pp]
                    int_t = basis_set[sigma_tt]

                    h_two[int_p, int_t] += h2[i][j][k][l]
                    h_two[int_t, int_p] += h2[k][l][i][j].conj()

            if k == j:
                op_il = (op[0], op[-1])
                sigma_qq = op_on_sigma(op_il, sigma_pp)

                if sigma_qq in basis_set.keys():
                    int_q = basis_set[sigma_qq]
                    int_p = basis_set[sigma_pp]

                    h_two[int_p, int_q] -= h2[i][j][k][l]
                    h_two[int_q, int_p] -= h2[k][l][i][j].conj()

    # Return the compact Hamiltonian H_c
    return h_one + h_two # + H_ferm.constant

def h_to_qubitop(h_c, n):

    qu_op = QubitOperator()

    for pauli_tensor in itertools.product("IXYZ", repeat=n):
        pauli_word = "".join(pauli_tensor)
        term = pauli_string_to_of(pauli_word)

        term_op = QubitOperator(term, 1.)

        c_j = np.trace(h_c.conj().T @ qubit_operator_sparse(term_op, n_qubits=n).todense())
        qu_op += QubitOperator(term, c_j)

    qu_op *= 1 / np.sqrt(2**n) * 0.5 # Why the 0.5 factor!!!
    qu_op.compress()
    return qu_op


if __name__ == "__main__":

    from openfermion.linalg import eigenspectrum
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g
    mol = mol_H2_sto3g

    H_ferm = chemist_ordered(mol.fermionic_hamiltonian)
    true_eigs = eigenspectrum(H_ferm)
    print(true_eigs)

    core_constant, one_body_integrals, two_body_integrals = mol.get_active_space_integrals()
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_body_integrals, two_body_integrals)

    H = compact_hamiltonian(H_ferm, mol.n_active_mos, mol.n_active_electrons, one_body_coefficients, two_body_coefficients)
    #norm_factor = np.trace(H.conj().T @ H)
    #H /= np.sqrt(norm_factor)
    #H *= 2
    eigs, eigvs = np.linalg.eigh(H)
    print(eigs)
    #print(eigvs)

    #Hq = h_to_qubitop(H, 2)
    #print(Hq)
    #matrix = qubit_operator_sparse(Hq).todense()
    #eigs, eigvs = np.linalg.eigh(matrix)
    #print(eigs)
