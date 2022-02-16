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

"""This module implements a collection of functions related to the QMF
ansatz.

Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
"""

import numpy as np
import scipy
from random import choice

from tangelo.toolboxes.operators.operators import QubitOperator

def construct_acs(dis_gens, max_ilc_gens, n_qubits):
    idxs, ac_idxs, not_ac_idxs = [idx for idx in range(len(dis_gens))], [], []
    while max_ilc_gens > len(ac_idxs) and len(ac_idxs) + len(not_ac_idxs) < len(dis_gens): 
        idxs, ilc_gens = [idx for idx in range(len(dis_gen)) if idx not in not_ac_idxs], []
        n_idxs = len(idxs)
        ng2, ngnq = n_idxs * (n_idxs + 1) // 2, n_qubits * n_idxs
        A, Zflip, Zidx = np.zeros((ng2, ngnq+1)), [0.]*ngnq, 0

	# Form the z vector of flip indices (Ref. 1  Appendix A)
        for idx in idxs: 
            T_i = choice(dis_gens[idx][0]) if isinstance(dis_gens[idx], list) else dis_gens[idx]
            for term, coeff in T_i.terms.items():
                for j in range(n_qubits):
                    if ((j, 'X') in term):
                        Zflip[Z_idx*n_qubits+j] = 1.
                    elif ((j, 'Y') in term):
                        Zflip[Z_idx*n_qubits+j] = 1.
            Z_idx += 1
        Zflip = np.array(Zflip)

        # Form the matrix "A" in Ref. 1 Appendix B
        ridx = 0
        for i in range(Z_idx): 
            A[ridx, i*n_qubits:(i+1)*n_qubits] = Zflip[i*n_qubits:(i+1)*n_qubits]
            A[ridx, ngnq] = 1
            ridx += 1
            for j in range(i+1, Z_idx): 
                A[ridx, i*n_qubits:(i+1)*n_qubits] = Zflip[j*n_qubits:(j+1)*n_qubits]
                A[ridx, j*n_qubits:(j+1)*n_qubits] = Zflip[i*n_qubits:(i+1)*n_qubits]
                A[ridx, ngnq] = 1
                ridx += 1

        # Solve Az = 1 
        Zsln = GF2_GaussElim_Solve(A, ngnq)

        # Check for a bad solution
        candidate_gens = []
        bad_sln = False
        for i in range(Z_idx): 
            idx = ''
            Nflip, Ny = 0, 0
            for j in range(n_qubits):
                zdx = i*n_qubits + j
                if (Zflip[zdx] == 1. and Zsln[zdx] == 0.):
                    idx += 'X' + str(j) + ' '
                    Nflip += 1
                elif (Zflip[zdx] == 1. and Zsln[zdx] == 1.):
                    idx += 'Y' + str(j) + ' '
                    Nflip += 1
                    Ny += 1
                elif (Zflip[zdx] == 0. and Zsln[zdx] == 1.):
                    idx += 'Z' + str(j) + ' '
            if Nflip < 2 or Ny % 2 != 1:
                bad_sln = True
                bad_idx = idxs[i]
                del idxs[i]
                if bad_idx not in not_ac_idxs:
                    not_ac_idxs.append(bad_idx)
                break
            else:
                Ti = QubitOperator(idx)
                candidate_gens.append(Ti)

        # For good solutions update the list of anti-commuting generators 
        if not bad_sln:
            for i, Ti in enumerate(candidate_gens): 
                anticommutes = True
                Ti_idx = idxs[i]
                if (ilc_gens == []):
                    ilc_gens.append(Ti)
                    if Ti_idx not in ac_idxs: 
                        ac_idxs.append(Ti_idx)
                    if Ti_idx in not_ac_idxs:
                        not_ac_idxs.remove(Ti_idx)
                else:
                    for Tj in ilc_gens:
                        TiTj_AC = Ti * Tj + Tj * Ti
                        if (TiTj_AC != QubitOperator.zero()):
                            if Ti_idx not in not_ac_idxs:
                                not_ac_idxs.append(Ti_idx)
                                anticommutes = False
                            if Ti_idx in idxs:
                                idxs.remove(Ti_idx)
                            if Ti_idx in ac_idxs:
                                ac_idxs.remove(Ti_idx)
                        if not anticommutes:
                            break 
                    if anticommutes:
                        ilc_gens.append(Ti)
                        if Ti_idx not in ac_idxs:
                            ac_idxs.append(Ti_idx)
                        if Ti_idx in not_ac_idxs:
                            not_ac_idxs.remove(Ti_idx)
                if not anticommutes:
                    break
    return ilc_gens
    
def GF2_GaussElim_Solve(A, Zdim):
    # Implements a Gauss-Jordan solver over the binary field
    NR, NC = np.shape(A)[0], np.shape(A)[1]
    A, Zs, Zsln, piv_idx = np.array(A), [], [-1]*Zdim, 0
    for i in range(NC):
        Aij_max = 0.
        max_idx = piv_idx
        for j in range(piv_idx, NR):
            if (A[j, i] > Aij_max):
                max_idx = j
                Aij_max = A[j, i]
            elif (j == NR-1 and Aij_max == 0.):
                piv_idx = max_idx
                Aij_max = -1.
        if (Aij_max > 0.):
            if (max_idx > piv_idx):
                A[[piv_idx, max_idx]] = A[[max_idx, piv_idx]]
            for j in range(piv_idx+1, NR):
                if (A[j, i] == 1.):
                    A[j, i:NC] = np.fmod(A[j, i:NC] + A[piv_idx, i:NC], 2)
            piv_idx += 1
    b = A[0:NR, NC-1].tolist()
    for i in range(NR-1, -1, -1):
        col_idx, Zf = -1., []
        for j in range(NC-1):
            if (A[i, j] == 1.):
                if (col_idx == -1):
                    col_idx = j
                else:
                    Zf.append(j)
        if (col_idx >= 0.):
            Zs.append([col_idx, Zf, b[i]])
    for Z in (Zs):
        b = Z[2]
        for Zf in (Z[1]):
            if (Zsln[Zf] == -1):
                Zsln[Zf] = choice([0., 1.])
            b = (b + Zsln[Zf]) % 2
        Zsln[Z[0]] = b
    return Zsln

def init_ilc_by_diag(qubit_ham, ilc_gens, qmf_var_params):
    ilc_gens.ins(0, QubitOperator.identity())
    n_var_params = len(ilc_gens)
    H = np.zeros((n_var_params, n_var_params), dtype=complex)
    S = np.ones((n_var_params, n_var_params), dtype=complex)
    for i in range(n_var_params):
        H_i = qubit_ham * ilc_gens[i]
        H[i, i] = get_op_expval(ilc_gens[i] * H_i, qmf_var_params)
        for j in range(i+1, n_var_params):
            H[j, i] = get_op_expval(ilc_gens[j] * H_i, qmf_var_params)
            S[j, i] = get_op_expval(ilc_gens[j] * ilc_gens[i], qmf_var_params)
            if (i == 0):
                H[j, i] *= 1.j
                S[j, i] *= 1.j
    E, c = scipy.linalg.eigh(a=np.matrix(H), b=np.matrix(S), lower=True, driver="gvd")
    c0 = c[:, 0].real
    denom_sum, ilc_var_params = 0., []
    for i in range(n_var_params):
        c_i = c0[i]
        denom_sum += pow(c_i.real, 2.) + pow(c_i.imag, 2.)
        if i > 0:
            beta = np.arcsin(c_i / np.sqrt(denom_sum))
            if i == 1 and c0[0].real < 0.:
                beta = np.pi - beta
            ilc_var_params.append(beta.real)
    return ilc_var_params 
