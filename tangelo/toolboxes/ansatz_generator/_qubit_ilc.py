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
"""

import warnings
from math import acos, sin, sqrt

import scipy
import numpy as np
from openfermion import commutator

from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator._qubit_mf import get_op_expval


def construct_acs(dis, n_qubits):
    """Driver function for constructing the anticommuting set of generators from
    the direct interaction set (DIS) of QCC generators.

    Args:
        dis (list of list): DIS of QCC generators.
        n_qubits (int): number of qubits

    Returns:
        list of QubitOperator: the anticommuting set (ACS) of ILC generators
    """

    bad_sln_idxs = []
    n_dis_groups = len(dis)

    good_sln = False
    while not good_sln:
        gen_idxs, ilc_gens = [idx for idx in range(n_dis_groups) if idx not in bad_sln_idxs], []
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
    over the binary field where b is the known solution vector. The elements
    of a_mat and b_vec need to be 0 or 1. If they are not provided as such,
    they will be transformed accordingly.

    Args:
        a_mat (numpy array of int): rectangular matrix of dimension n x m that
            holds the action of A * z, where z is a column vector of dimension m x 1.
            No default.
        b_vec (numpy array of int): column vector of dimension n x 1 holding the
            initial solution of A * z. Default, np.zeros((n, 1)).

    Returns:
        numpy array of int: solution for the z vector of dimension (n, )
    """

    # check that b_vec was provided properly; otherwise initialize as a vector of zeros
    n_rows, n_cols = np.shape(a_mat)
    if not isinstance(b_vec, np.ndarray):
        b_vec = np.zeros((n_rows, 1))

    # append the initial solution vector as the last column of a_mat; update n_cols
    a_mat = np.concatenate((a_mat, b_vec), axis=1).astype('int8')
    n_cols += 1

    # force all entries of a_mat to be either 0 or 1.
    if (abs(a_mat) > 1).any():
        warnings.warn("Reducing input matrix elements modulo 2 to create a binary matrix.", RuntimeWarning)
        a_mat = (a_mat % 2).astype('int8')

    # remove duplicate rows if they exist
    _, row_idxs = np.unique([tuple(row) for row in a_mat], axis=0, return_index=True)
    a_mat_new = a_mat[np.sort(row_idxs)]
    if a_mat_new.shape[0] != a_mat.shape[0]:
        warnings.warn("Linear dependency detected in input matrix: redundant rows deleted.", RuntimeWarning)
    a_mat = a_mat_new

    # remove rows of all 0s if they exist
    del_idxs = []
    for i in range(a_mat.shape[0]):
        if (a_mat[i][:] == 0).all():
            del_idxs.append(i)
    if del_idxs:
        warnings.warn("Linear dependency detected in input matrix: rows of zeros deleted.", RuntimeWarning)
    a_mat = np.delete(a_mat, obj=del_idxs, axis=0)
    n_rows = a_mat.shape[0]

    # begin gaussian elimination algorithm
    z_sln, piv_idx = [-1]*(n_cols-1), 0
    for i in range(n_cols):
        a_mat_max, max_idx = 0, piv_idx
        # locate the pivot index by searching each row for a non-zero value
        for j in range(piv_idx, n_rows):
            # if a pivot index is found, mark the value of the column index
            if a_mat[j, i] > a_mat_max:
                max_idx = j
                a_mat_max = a_mat[j, i]
            # if a pivot index is not found, reset and move to the next row
            elif j == n_rows-1 and a_mat_max == 0:
                piv_idx = max_idx
                a_mat_max = -1
        # update the matrix by flipping the row and columns to achieve row echelon form
        if a_mat_max > 0:
            if max_idx > piv_idx:
                a_mat[[piv_idx, max_idx]] = a_mat[[max_idx, piv_idx]]
            for j in range(piv_idx + 1, n_rows):
                if a_mat[j, i] == 1:
                    a_mat[j, i:n_cols] = np.fmod(a_mat[j, i:n_cols] + a_mat[piv_idx, i:n_cols], 2)
            piv_idx += 1

    # extract the solution: back solve from the last row to the first
    for i in range(n_rows-1, -1, -1):
        # find the first non-zero coefficient for a row
        j = 0
        while a_mat[i, j] == 0 and j < n_cols-2:
            j += 1
        # initialize the soln then back solve
        b_val = a_mat[i, n_cols-1]
        if i == n_rows-1 and j == n_cols-2:
            z_sln[j] = int(b_val)
        else:
            for k in range(j+1, n_cols-1):
                if a_mat[i, k] == 1:
                    if z_sln[k] == -1:
                        z_sln[k] = 0
                    else:
                        b_val = (b_val + z_sln[k]) % 2
            z_sln[j] = int(b_val)

    # set any remaining free variables to 0
    for i, sln_val in enumerate(z_sln):
        if sln_val == -1:
            z_sln[i] = 0
    return np.array(z_sln, dtype=int)


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

        for j in range(i+1, n_var_params):
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
        denom_sum += abs(gs_coefs[i])**2
    beta_1 = np.arcsin(gs_coefs[1] / np.sqrt(denom_sum))
    ilc_var_params.append(beta_1.real)
    for i in range(2, n_var_params):
        denom_sum += abs(gs_coefs[i])**2
        beta = np.arcsin(gs_coefs[i] / np.sqrt(denom_sum))
        ilc_var_params.append(beta.real)
    del ilc_gens[0]
    return ilc_var_params


def build_ilc_qubit_op_list(acs_gens, ilc_params):
    """Returns a list of 2N - 1 ILC generators to facilitate generation of a circuit
    based on symmetric Trotter-Suzuki decomposition. The ILC generators are ordered
    according to Eq. C1 in Appendix C of Ref. 1.

    Args:
        acs_gens (list of QubitOperator): The list of ILC Pauli word generators
            selected from characteristic ACS groups.
        ilc_params (list or numpy array of float): The ILC variational parameters
            arranged such that their ordering matches the order of acs_gens.

    Returns:
        list of QubitOperator: list of ILC ansatz operator generators.
    """

    n_amps = len(ilc_params)
    ilc_op_list = [-.5 * ilc_params[i] * acs_gens[i] for i in range(n_amps-1, 0, -1)]
    ilc_op_list += [ilc_params[0] * acs_gens[0]]
    ilc_op_list += [-.5 * ilc_params[i] * acs_gens[i] for i in range(1, n_amps)]
    return ilc_op_list


def ilc_op_dress(qubit_op, ilc_gens, ilc_params):
    """Performs transformation of a qubit operator with the ACS of ILC generators and
    parameters. For a set of N generators, each qubit operator transformation results
    in quadratic (N * (N-1) / 2) growth of the number of its terms.

    Args:
        qubit_op (QubitOperator): A qubit operator to be dressed.
        ilc_gens (list of QubitOperator): The list of ILC Pauli word generators
            selected from a user-specified number of characteristic ACS groups.
        ilc_params (list or numpy array of float): The ILC variational parameters
            arranged such that their ordering matches the ordering of ilc_gens.

    Returns:
        QubitOperator: Dressed qubit operator.
    """

    # first, recast the beta parameters into the set of coefficients {c_n}
    n_amps = len(ilc_params)
    coef_norm = 1.
    coefs = [0.] * n_amps

    # See Ref. 1, Appendix C, Eqs. C3 and C4:
    # sin b_n = c_n; sin_b_n-1 = c_n-1 / sqrt(1-|c_n|**2);
    # sin_b_n-2 = c_n-2 / sqrt(1-|c_n|**2-|c_n-1|**2) ...
    for i in range(n_amps-1, -1, -1):
        coef = sqrt(coef_norm) * sin(ilc_params[i])
        coefs[i] = coef
        coef_norm -= coef**2

    # the remainder of coef_norm is |c_0|^2
    coefs.insert(0, -sqrt(coef_norm))

    # second, recast {c_n} into tau, {alpha_i};
    # c_0 = cos(tau); c_n = sin(tau) * alpha_n for n > 0
    tau = acos(coefs[0])
    sin_tau = sin(tau)
    alphas = [coefs[i]/sin_tau for i in range(1, n_amps+1)]

    # third, dress the qubit operator according to Eqs. 17, 18 in Ref. 2
    sin2_tau = sin_tau**2
    sin_2tau = sin(2.*tau)
    qop_dress = coefs[0]**2 * qubit_op
    for i in range(n_amps):
        qop_dress += sin2_tau * alphas[i]**2 * ilc_gens[i] * qubit_op * ilc_gens[i]\
                  - .5j * sin_2tau * alphas[i] * commutator(qubit_op, ilc_gens[i])
        for j in range(i+1, n_amps):
            qop_dress += sin2_tau * alphas[i] * alphas[j] * (ilc_gens[i] * qubit_op * ilc_gens[j]
                      + ilc_gens[j] * qubit_op * ilc_gens[i])
    qop_dress.compress()
    return qop_dress
