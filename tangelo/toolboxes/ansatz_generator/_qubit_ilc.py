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

"""This module implements a collection of functions related to the ILC ansatz:
1. Function to create the anticommuting set (ACS) of generators from the QCC DIS;
2. An efficient solver that performs Gaussian elimination over GF(2);
3. Function that computes the ILC parameters via matrix diagonalization.

Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        arXiv:2002.05701v1, 2020, 1–10.
    2. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
    3. Ç. K. Koç and S. N. Arachchige.
        J. Parallel Distrib. Comput., 1991, 13, 118–122.
"""

import warnings

import scipy
import numpy as np

from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator._qubit_mf import get_op_expval


def construct_acs(dis, max_ilc_gens, n_qubits):
    """Driver function for constructing the anticommuting set of generators from
    the direct interaction set (DIS) of QCC generators.

    Args:
        dis (list of list): DIS of QCC generators.
        max_ilc_gens (int): maximum number of ILC generators allowed in the ansatz.
        n_qubits (int): number of qubits

    Returns:
        list of QubitOperator: the anticommuting set (ACS) of ILC generators
    """

    bad_sln_idxs, good_sln = [], False
    while not good_sln:
        gen_idxs, ilc_gens = [idx for idx in range(max_ilc_gens) if idx not in bad_sln_idxs], []
        n_gens = len(gen_idxs)
        ng2, ngnq = n_gens * (n_gens + 1) // 2, n_gens * n_qubits

        # a_mat --> A and z_vec --> z in Appendix A, Refs. 1 & 2.
        a_mat, z_vec, one_vec = np.zeros((ng2, ngnq)), np.zeros(ngnq), np.ones((ng2, 1))
        for idx, gen_idx in enumerate(gen_idxs):
            gen = dis[gen_idx]
            for term in gen.terms:
                for paulis in term:
                    p_idx, pauli = paulis
                    if 'X' in pauli or 'Y' in pauli:
                        z_vec[idx * n_qubits + p_idx] = 1.

        # Form the rectangular matrix-vector product A * z (Appendix A, Refs. 1 & 2).
        rowdx = 0
        for i in range(n_gens):
            a_mat[rowdx, i * n_qubits:(i+1) * n_qubits] = z_vec[i * n_qubits:(i+1) * n_qubits]
            rowdx += 1
            for j in range(i + 1, n_gens):
                a_mat[rowdx, i * n_qubits:(i+1) * n_qubits] = z_vec[j * n_qubits:(j+1) * n_qubits]
                a_mat[rowdx, j * n_qubits:(j+1) * n_qubits] = z_vec[i * n_qubits:(i+1) * n_qubits]
                rowdx += 1

        # Solve A * z = b --> here b = 1
        z_sln = gauss_elim_over_gf2(a_mat, b_vec=one_vec)

        # Check solution: odd # of Y ops, at least two flip indices, and mutually anticommutes
        for i in range(n_gens):
            n_flip, n_y, gen_idx, gen_tup = 0, 0, gen_idxs[i], tuple()
            for j in range(n_qubits):
                gen, idx = None, i * n_qubits + j
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
                    gen_tup += (gen, )
            # check number of flip indices and number of Y Pauli ops
            if n_flip > 1 and n_y % 2 == 1:
                gen_i = QubitOperator(gen_tup, 1.)
                good_sln = True
                # check mutual anticommutativity of each new ILC generator with all the rest
                for gen_j in ilc_gens:
                    if gen_i * gen_j != -1. * gen_j * gen_i:
                        if gen_idx not in bad_sln_idxs:
                            bad_sln_idxs.append(gen_idx)
                            good_sln = False
            else:
                if gen_idx not in bad_sln_idxs:
                    bad_sln_idxs.append(gen_idx)
                    good_sln = False
            if good_sln:
                ilc_gens.append(gen_i)
    return ilc_gens


def gauss_elim_over_gf2(a_mat, b_vec=None):
    """Driver function that performs Gaussian elimination to solve A * z = b
    over the binary field where b is the known solution vector. This routine
    was adapted based on Ref. 3. All elements of a_mat and b_vec are assumed
    to be the integers 0 or 1.

    Args:
        a_mat (numpy array of int): rectangular matrix of dimension n x m that
            holds the action of A * z, where z is a column vector of dimension m x 1.
            No default.
        b_vec (numpy array of int): column vector of dimension n x 1 holding the
            initial solution of A * z. Default, np.zeros((n, 1)).

    Returns:
        numpy array of float: solution for the z vector of dimension (n, )
    """

    n_rows, n_cols = np.shape(a_mat)
    z_vals, z_sln, piv_idx = [], [-1] * n_cols, 0
    # check that b_vec was properly supplied; ortherwise initialize as a vector of zeros
    if not isinstance(b_vec, np.ndarray):
        b_vec = np.zeros((n_rows, 1))
    a_mat = np.append(a_mat, b_vec, axis=1)
    n_cols += 1
    for i in range(n_cols):
        a_mat_max, max_idx = 0., piv_idx
        # locate the pivot index by searching each row for a non-zero value.
        for j in range(piv_idx, n_rows):
            # if a pivot index is found, set the value to the col index for the row in which it was found
            if a_mat[j, i] > a_mat_max:
                max_idx = j
                a_mat_max = a_mat[j, i]
            # if a pivot index is not found in a given row, reset a_mat_max to -1 and move to the next row
            elif j == n_rows-1 and a_mat_max == 0.:
                piv_idx = max_idx
                a_mat_max = -1.
        # update the matrix by flipping the row and columns to achieve row echelon form
        if a_mat_max > 0.:
            if max_idx > piv_idx:
                a_mat[[piv_idx, max_idx]] = a_mat[[max_idx, piv_idx]]
            for j in range(piv_idx + 1, n_rows):
                if a_mat[j, i] == 1.:
                    a_mat[j, i:n_cols] = np.fmod(a_mat[j, i:n_cols] + a_mat[piv_idx, i:n_cols], 2)
            piv_idx += 1
    # extract the solution from the bottom to the top since it is now in row echelon form
    b_vec = a_mat[0:n_rows, n_cols-1].tolist()
    for i in range(n_rows - 1, -1, -1):
        col_idx, z_free = -1., []
        for j in range(n_cols-1):
            if a_mat[i, j] == 1.:
                if col_idx == -1:
                    col_idx = j
                else:
                    z_free.append(j)
        if col_idx >= 0.:
            z_vals.append([col_idx, z_free, b_vec[i]])
    # check for free solutions -- select 0 for the free solution
    # for the ILC generator screening procedure, 0 leads to an I op and 1 leads to a Z Pauli op
    for z_val in (z_vals):
        b_val = z_val[2]
        for z_free in (z_val[1]):
            if z_sln[z_free] == -1:
                z_sln[z_free] = 0.
            b_val = np.fmod(b_val + z_sln[z_free], 2)
        z_sln[z_val[0]] = b_val
    # check that z_sln does not have any -1 values left -- if so, a solution was not found.
    for z_val in z_sln:
        if z_val == -1:
            warnings.warn("Gaussian elimination over GF(2) failed to find a solution.", RuntimeWarning)
    return np.array(z_sln)


def get_ilc_params_by_diag(qubit_ham, ilc_gens, qmf_var_params):
    """Driver function that solves the generalized eigenvalue problem Hc = ESc required
    to obtain the ground state coefficients (ILC parameters). These are subsequently recast
    according to Appendix C of Ref. 1 in a form that is suitable for constructing ILC circuits.

    Args:
        qubit_ham (QubitOperator): the qubit Hamiltonian of the system.
        ilc_gens (list of QubitOperator): the anticommuting set of ILC Pauli words.

    Returns:
        list of float: the ILC parameters corresponding to the ACS of ILC generators
    """

    # Add the identity operator to the local copy of the ACS
    ilc_gens.insert(0, QubitOperator.identity())
    n_var_params = len(ilc_gens)
    qubit_ham_mat = np.zeros((n_var_params, n_var_params), dtype=complex)
    qubit_overlap_mat = np.zeros((n_var_params, n_var_params), dtype=complex)

    # Construct the lower triangular matrices for the qubit Hamiltonian and overlap integrals
    for i in range(n_var_params):
        # H T_i|QMF> = H |psi_i>
        h_psi_i = qubit_ham * ilc_gens[i]

        # <QMF|T_i H T_i|QMF> = <psi_i| H | psi_i> = H_ii
        qubit_ham_mat[i, i] = get_op_expval(ilc_gens[i] * h_psi_i, qmf_var_params)

        # <QMF|T_i T_i|QMF> = <psi_i|psi_i> = 1
        qubit_overlap_mat[i, i] = 1. + 0j

        for j in range(i + 1, n_var_params):
            # <QMF|T_j H T_i|QMF> = <psi_j| H | psi_i> = H_ji
            qubit_ham_mat[j, i] = get_op_expval(ilc_gens[j] * h_psi_i, qmf_var_params)

            # <QMF|T_j T_i|QMF> = <psi_j|psi_i> --> exactly zero only for pure QMF states
            qubit_overlap_mat[j, i] = get_op_expval(ilc_gens[j] * ilc_gens[i], qmf_var_params)
            if i == 0:
                qubit_ham_mat[j, i] *= 1j
                qubit_overlap_mat[j, i] *= 1j

    # Solve the generalized eigenvalue problem
    _, subspace_coefs = scipy.linalg.eigh(a=qubit_ham_mat, b=qubit_overlap_mat, lower=True, driver="gvd")

    # Compute the ILC parameters using the ground state coefficients
    gs_coefs = subspace_coefs[:, 0]
    if gs_coefs[0].real > 0.:
        gs_coefs *= -1.
    denom_sum, ilc_var_params = 0., []
    for i in range(2):
        denom_sum += pow(gs_coefs[i].real, 2.) + pow(gs_coefs[i].imag, 2.)
    beta_1 = np.arcsin(gs_coefs[1] / np.sqrt(denom_sum))
    ilc_var_params.append(beta_1.real)
    for i in range(2, n_var_params):
        denom_sum += pow(gs_coefs[i].real, 2.) + pow(gs_coefs[i].imag, 2.)
        beta = np.arcsin(gs_coefs[i] / np.sqrt(denom_sum))
        ilc_var_params.append(beta.real)
    del ilc_gens[0]
    return ilc_var_params
