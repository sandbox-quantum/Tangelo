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

"""This module implements a collection of functions related to the ILC
ansatz:
    1. Function to create the anti-commuting set (ACS) of generators from
       the QCC DIS;
    2. An efficient solver that performs Gaussian elimination over GF(2);
    3. Function that initializes the ILC parameters via matrix diagonalization.

Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        arXiv:2002.05701v1, 2020, 1–10.
    2. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
    3. Ç. K. Koç and S. N. Arachchige.
        J. Parallel Distrib. Comput., 1991, 13, 118–122.
"""

from random import choice
import scipy
import numpy as np

from tangelo.toolboxes.operators.operators import QubitOperator
from ._qubit_mf import get_op_expval


def construct_acs(dis_gens, max_ilc_gens, n_qubits):
    ac_idxs, not_ac_idxs = [], []
    while max_ilc_gens > len(ac_idxs) and len(ac_idxs) + len(not_ac_idxs) < max_ilc_gens:
        gen_idxs, ilc_gens = [idx for idx in range(max_ilc_gens) if idx not in not_ac_idxs], []
        n_gens = len(gen_idxs)
        ng2, ngnq = n_gens * (n_gens + 1) // 2, n_gens * n_qubits
        # cons_mat --> A and z_vec --> z in Appendix A, Refs. 1 & 2.
        cons_matrix, z_vec = np.zeros((ng2, ngnq + 1)), np.zeros(ngnq)
        for idx, gen_idx in enumerate(gen_idxs):
            gen = dis_gens[gen_idx][-1]
            for term, _ in gen.terms.items():
                for paulis in term:
                    p_idx, pauli = paulis
                    if 'X' in pauli or 'Y' in pauli:
                        z_vec[idx * n_qubits + p_idx] = 1.

        # Form the triangular matrix A (Appendix A, Refs. 1 & 2).
        r_idx = 0
        for i in range(n_gens):
            cons_matrix[r_idx, i*n_qubits:(i+1)*n_qubits] = z_vec[i*n_qubits:(i+1)*n_qubits]
            cons_matrix[r_idx, ngnq] = 1
            r_idx += 1
            for j in range(i+1, n_gens):
                cons_matrix[r_idx, i*n_qubits:(i+1)*n_qubits] = z_vec[j*n_qubits:(j+1)*n_qubits]
                cons_matrix[r_idx, j*n_qubits:(j+1)*n_qubits] = z_vec[i*n_qubits:(i+1)*n_qubits]
                cons_matrix[r_idx, ngnq] = 1
                r_idx += 1

        # Solve Az = 1
        z_sln = gauss_elim_over_gf2(cons_matrix, ngnq)

        # Check for a bad solutions
        candidate_gens, good_sln = [], True
        for i in range(n_gens):
            n_flip, n_y, gen_list = 0, 0, []
            for j in range(n_qubits):
                gen = None
                idx = i * n_qubits + j
                if z_vec[idx] == 1.:
                    n_flip += 1
                    if z_sln[idx] == 0.:
                        gen = (j, 'X')
                    else:
                        gen = (j, 'Y')
                        n_y += 1
                else:
                    if z_sln[idx] == 1.:
                        gen = (j, 'Z')
                if gen:
                    gen_list.append(gen)
            if n_flip < 2 or n_y % 2 == 0:
                good_sln = False
                gen_idx = gen_idxs.pop(i)
                if gen_idx not in not_ac_idxs:
                    not_ac_idxs.append(gen_idx)
                if gen_idx in gen_idxs:
                    gen_idxs.remove(gen_idx)
                if gen_idx in ac_idxs:
                    ac_idxs.remove(gen_idx)
            else:
               candidate_gens.append(QubitOperator(tuple(gen_list), 1.))

        # For good solutions check that they anti-commute and update ilc_gens
        if good_sln:
            for i, gen_i in enumerate(candidate_gens):
                anticommutes = True
                gen_idx = gen_idxs[i]
                for gen_j in ilc_gens:
                    anti_com = gen_i * gen_j + gen_j * gen_i
                    if anti_com != QubitOperator.zero():
                        anticommutes = False
                        if gen_idx not in not_ac_idxs:
                            not_ac_idxs.append(gen_idx)
                        if gen_idx in gen_idxs:
                            gen_idxs.remove(gen_idx)
                        if gen_idx in ac_idxs:
                            ac_idxs.remove(gen_idx)
                if anticommutes:
                    ilc_gens.append(gen_i)
                    if gen_idx not in ac_idxs:
                        ac_idxs.append(gen_idx)
                    if gen_idx in not_ac_idxs:
                        not_ac_idxs.remove(gen_idx)
    return ilc_gens


def gauss_elim_over_gf2(A, zdim):
    # Gaussian elimination over GF(2) -- based on Ref. 3.
    n_rows, n_cols = np.shape(A)[0], np.shape(A)[1]
    A, zs, z_sln, piv_idx = np.array(A), [], [-1]*zdim, 0
    for i in range(n_cols):
        Aij_max = 0.
        max_idx = piv_idx
        for j in range(piv_idx, n_rows):
            if A[j, i] > Aij_max:
                max_idx = j
                Aij_max = A[j, i]
            elif j == n_rows-1 and Aij_max == 0.:
                piv_idx = max_idx
                Aij_max = -1.
        if Aij_max > 0.:
            if max_idx > piv_idx:
                A[[piv_idx, max_idx]] = A[[max_idx, piv_idx]]
            for j in range(piv_idx+1, n_rows):
                if A[j, i] == 1.:
                    A[j, i:n_cols] = np.fmod(A[j, i:n_cols] + A[piv_idx, i:n_cols], 2)
            piv_idx += 1
    b = A[0:n_rows, n_cols-1].tolist()
    for i in range(n_rows-1, -1, -1):
        col_idx, zf = -1., []
        for j in range(n_cols-1):
            if A[i, j] == 1.:
                if col_idx == -1:
                    col_idx = j
                else:
                    zf.append(j)
        if col_idx >= 0.:
            zs.append([col_idx, zf, b[i]])
    for z in (zs):
        b = z[2]
        for zf in (z[1]):
            if z_sln[zf] == -1:
                z_sln[zf] = choice([0., 1.])
            b = (b + z_sln[zf]) % 2
        z_sln[z[0]] = b
    return z_sln


def init_ilc_by_diag(qubit_ham, ilc_gens, qmf_var_params):
    ilc_gens.insert(0, QubitOperator.identity())
    n_var_params = len(ilc_gens)
    # Form the Hamiltonian and overlap matrices (see Appendix B, Refs. 1 & 2).
    H = np.zeros((n_var_params, n_var_params), dtype=complex)
    S = np.zeros((n_var_params, n_var_params), dtype=complex)
    for i in range(n_var_params):
        H_i = qubit_ham * ilc_gens[i]
        H[i, i] = get_op_expval(ilc_gens[i] * H_i, qmf_var_params)
        S[i, i] = 1. + 0j
        for j in range(i + 1, n_var_params):
            H[j, i] = get_op_expval(ilc_gens[j] * H_i, qmf_var_params)
            S[j, i] = get_op_expval(ilc_gens[j] * ilc_gens[i], qmf_var_params)
            if i == 0:
                H[j, i] *= 1j
                S[j, i] *= 1j

    # Solve the generalized eigenvalue problem
    E, c = scipy.linalg.eigh(a=np.matrix(H), b=np.matrix(S), lower=True, driver="gvd")
    print(" MCSCF eigenvalues from matrix diagonalization = ", E)

    # Compute the ILC parameters according to Appendix C, Ref. 1).
    c0 = c[:, 0]
    print(" Ground state eigenvector = ", c0)
    denom_sum, ilc_var_params = 0., []
    for i in range(2):
        denom_sum += pow(c0[i].real, 2.) + pow(c0[i].imag, 2.)
    beta_1 = np.arcsin(c0[1] / np.sqrt(denom_sum))
    if c0[0].real > 0.:
        beta_1 = np.pi - beta_1
    ilc_var_params.append(beta_1.real)
    for i in range(2, n_var_params):
        denom_sum += pow(c0[i].real, 2.) + pow(c0[i].imag, 2.)
        beta = np.arcsin(c0[i] / np.sqrt(denom_sum))
        ilc_var_params.append(beta.real)
    del ilc_gens[0]
    print(" ILC var params (beta's in Appendix C, ref. 1) = ", ilc_var_params)
    return ilc_var_params
