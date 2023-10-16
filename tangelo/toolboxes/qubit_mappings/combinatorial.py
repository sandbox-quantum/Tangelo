# Copyright 2023 Good Chemistry Company.
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

"""Combinatorial mapping as described in references (1) and (2). In contrast to
qubit mappings such as JW, BK that use occupation/parity, the mappings in this
file use the Fock configurations as the elements of the basis set. Thus, the
number of required qubits scales with the number of electronic configuration
instead of the number of spinorbitals.

References:
    1. Streltsov, A. I., Alon, O. E. & Cederbaum, L. S. General mapping for
        bosonic and fermionic operators in Fock space. Phys. Rev. A 81, 022124
        (2010).
    2. Chamaki, D., Metcalf, M. & de Jong, W. A. Compact Molecular Simulation on
        Quantum Computers via Combinatorial Mapping and Variational State
        Preparation. Preprint at https://doi.org/10.48550/arXiv.2205.11742
        (2022).
"""

import sys
import itertools
import math
from collections import OrderedDict
from functools import lru_cache
from math import ceil

import numpy as np
from scipy.special import comb
from openfermion.transforms import chemist_ordered
import gc


from tangelo.toolboxes.operators import QubitOperator


ZERO_TOLERANCE = 1e-8

#@profile
def int_to_tuple(integer, n_qubits):

    if integer == 131072:
        pass

    term = []
    for i in range(1, n_qubits+1):
        shift_x = 2*(i-1)
        shift_z = shift_x + 1

        x_term = (integer & (1 << shift_x)) >> shift_x
        z_term = (integer & (1 << shift_z)) >> shift_z

        if (x_term, z_term) == (0, 0):
            continue
        elif (x_term, z_term) == (1, 0):
            term.append((i-1, 'X'))
        elif (x_term, z_term) == (0, 1):
            term.append((i-1, 'Z'))
        else:
            term.append((i-1, 'Y'))

    return tuple(term)


def tensor_product_pauli_dicts(pa, pb):
    pp = dict()
    for ta, ca in pa.items():
        for tb, cb in pb.items():
            pp[ta ^ tb] = ca*cb
    return pp


@lru_cache()
def prep(n):
    # print(n)
    n_qubits = int(math.log2(n))
    shift_x = 2 * (n_qubits - 1)
    shift_z = shift_x + 1

    # 1/2 (I +-Z)
    z_op = (1 << shift_z)
    i_plus_z = {0: 0.5, z_op: 0.5}
    i_minus_z = {0: 0.5, z_op: -0.5}

    # 1/2 (X +-iY)
    x_op = (1 << shift_x)
    y_op = x_op | (1 << shift_z)
    x_plus_iy = {x_op: 0.5, y_op: 0.5j}
    x_minus_iy = {x_op: 0.5, y_op: -0.5j}

    return i_plus_z, i_minus_z, x_plus_iy, x_minus_iy

#@profile
def combinatorial4(ferm_op, n_modes, n_electrons):

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f'{n=:}')

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = dict()
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[(unique_int, new_unique_int)] = quop_matrix.get((unique_int, new_unique_int), 0.) + phase*coeff

    # Valentin: quop is a Dict[(int, int) -> complex]
    print(f'combinatorial4 :: quop dict size {len(quop_matrix)} \t (memory :: {sys.getsizeof(quop_matrix)} bytes)')
    return QubitOperator()

    # Converts matrix back into qubit operator object
    gsize = 2**n # total size
    get_tensor_ops = {2**k: prep(2**k) for k in range(1, n+1)}

    def compute2(tM):
        m00, m01, m10, m11 = tM.get((0, 0), 0.), tM.get((0, 1), 0.), tM.get((1, 0), 0.), tM.get((1, 1), 0.)
        res = {0: 0.5 * (m00 + m11), 1: 0.5 * (m01 + m10), 2: 0.5 * (m00 - m11), 3: 0.5j * (m01 - m10)}
        return res

    # Split data across all 2x2 matrices
    from itertools import product
    t = dict()
    t[2] = {(xl, yl): dict() for (xl, yl) in product(range(0, 2**(n-1)), range(0, 2**(n-1)))}
    t[4] = {(xl, yl): dict() for (xl, yl) in product(range(0, 2**(n-2)), range(0, 2**(n-2)))}

    for ((x, y), v) in quop_matrix.items():
        xl, yl, xr, yr = x//2, y//2, x%2, y%2
        t[2][(xl, yl)][(xr, yr)] = v
    quop_matrix.clear() #del quop_matrix; gc.collect()

    # Agglomerate to level 4
    ops = get_tensor_ops[4]
    for (x, y) in product(range(0, 2**(n-1), 2), range(0, 2**(n-1), 2)):

        M_00 = tensor_product_pauli_dicts(compute2(t[2][(x,y)]), ops[0])
        M_11 = tensor_product_pauli_dicts(compute2(t[2][(x+1,y+1)]), ops[1])
        M_01 = tensor_product_pauli_dicts(compute2(t[2][(x,y+1)]), ops[2])
        M_10 = tensor_product_pauli_dicts(compute2(t[2][(x+1,y)]), ops[3])

        for (k, v) in M_01.items(): M_00[k] = M_00.get(k, 0.) + v
        for (k, v) in M_10.items(): M_00[k] = M_00.get(k, 0.) + v
        for (k, v) in M_11.items(): M_00[k] = M_00.get(k, 0.) + v

        xl, yl = x // 2, y // 2
        t[4][(xl, yl)] = M_00.copy()

    # Drop previous level out of memory, initialize iterative aggregation loop
    t[2].clear() #del t[2]; gc.collect()
    l, s = 8, 2**(n-2)

    while l <= gsize:
        t[l] = {(xl, yl): dict() for (xl, yl) in product(range(0, s//2), range(0, s//2))}
        ops = get_tensor_ops[l]
        for (x, y) in product(range(0, s, 2), range(0, s, 2)):

            M_00 = tensor_product_pauli_dicts(t[l//2][(x, y)], ops[0])
            M_11 = tensor_product_pauli_dicts(t[l//2][(x + 1, y + 1)], ops[1])
            M_01 = tensor_product_pauli_dicts(t[l//2][(x, y + 1)], ops[2])
            M_10 = tensor_product_pauli_dicts(t[l//2][(x + 1, y)], ops[3])

            for (k, v) in M_01.items(): M_00[k] = M_00.get(k, 0.) + v
            for (k, v) in M_10.items(): M_00[k] = M_00.get(k, 0.) + v
            for (k, v) in M_11.items(): M_00[k] = M_00.get(k, 0.) + v

            xl, yl = x // 2, y // 2
            t[l][(xl, yl)] = M_00.copy()

        # Next iteration
        t[l//2].clear() #del t[l//2]; gc.collect()
        l, s = 2 * l, s // 2

    quop_ints = t[l//2][(0,0)]

    # Construct final operator
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            t = int_to_tuple(term, n)
            quop.terms[t] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop


#@profile
def recursive_mapping_sparse(M, n, s): # n is n_rows and n_cols here

    # if n==4:
    #     print()

    #print(n)#, min(M.values()), max(M.values()))
    # Bottom of recursion: 2x2 matrix case
    if n == 2:
        M2 = {(k[0]%2, k[1]%2): v for k, v in M.items()}
        m00, m01, m10, m11 = M2.get((0, 0), 0.), M2.get((0, 1), 0.), M2.get((1, 0), 0.), M2.get((1, 1), 0.)
        res = {0: 0.5 * (m00 + m11), 1: 0.5 * (m01 + m10), 2: 0.5 * (m00 - m11), 3: 0.5j * (m01 - m10)}
        #print(res)
        return res
    else:

        #     #print(n)
        n_qubits = int(math.log2(n))
        shift_x = 2 * (n_qubits - 1)
        shift_z = shift_x + 1

        # 1/2 (I +-Z)
        z_op = (1 << shift_z)
        i_plus_z = {0: 0.5, z_op: 0.5}
        i_minus_z = {0: 0.5, z_op: -0.5}

        # 1/2 (X +-iY)
        x_op = (1 << shift_x)
        y_op = x_op | (1 << shift_z)
        x_plus_iy = {x_op: 0.5, y_op: 0.5j}
        x_minus_iy = {x_op: 0.5, y_op: -0.5j}

        piv = n // 2

        # Split into smaller dicts
        Ms_00, Ms_01, Ms_10, Ms_11 = dict(), dict(), dict(), dict()

        for ((x, y),v) in M.items():
            if x < piv + s[0]:
                if y < piv + s[1]:
                    Ms_00[(x, y)] = v
                else:
                    Ms_01[(x, y)] = v
            else:
                if y < piv + s[1]:
                    Ms_10[(x, y)] = v
                else:
                    Ms_11[(x, y)] = v

        if n == 8:
            pass

        M_00 = tensor_product_pauli_dicts(recursive_mapping_sparse(Ms_00, n//2, (s[0], s[1])), i_plus_z)
        M_11 = tensor_product_pauli_dicts(recursive_mapping_sparse(Ms_11, n//2,(s[0]+piv, s[1]+piv)), i_minus_z)
        M_01 = tensor_product_pauli_dicts(recursive_mapping_sparse(Ms_01, n//2, (s[0], s[1]+piv)), x_plus_iy)
        M_10 = tensor_product_pauli_dicts(recursive_mapping_sparse(Ms_10, n//2, (s[0]+piv, s[1])), x_minus_iy)

        # Merge the 4 outputs into one additively
        # d = dict()
        # for k in set(M_00.keys()) | set(M_01.keys()) | set(M_10.keys()) | set(M_11.keys()):
        #     # d[k] = sum(Mx.get(k, 0.) for Mx in [M_00, M_01, M_10, M_11])
        #     d[k] = M_00.get(k, 0) + M_01.get(k, 0) + M_10.get(k, 0) + M_11.get(k, 0)  # faster

        for (k,v) in M_01.items():
            M_00[k] = M_00.get(k, 0.) + v
        for (k,v) in M_10.items():
            M_00[k] = M_00.get(k, 0.) + v
        for (k,v) in M_11.items():
            M_00[k] = M_00.get(k, 0.) + v
        return M_00

#@profile
def combinatorial3(ferm_op, n_modes, n_electrons):

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f'{n=:}')

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = dict()
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one
    max_int = max(ints)
    n_terms = len(ferm_op_chemist.terms)

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[(unique_int, new_unique_int)] = quop_matrix.get((unique_int, new_unique_int), 0.) + phase*coeff

    # Valentin: is quop_matrix sparse ? Find % of non-zero elements
    print(f'combinatorial3 :: quop dict size {len(quop_matrix)} \t (memory :: {sys.getsizeof(quop_matrix)} bytes)')

    # Converts matrix back into qubit operator object
    quop_ints = recursive_mapping_sparse(quop_matrix, 2**n, (0,0))
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            t = int_to_tuple(term, n)
            quop.terms[t] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop

#@profile
def recursive_mapping(M):
    n_rows, n_cols = M.shape
    assert(n_rows == n_cols) # Shouldn't that be guaranteed if our code was done correctly ?

    # if n_rows==4:
    #     print()

    # Bottom of recursion: 2x2 matrix case
    #print(n_rows)#, np.min(M), np.max(M))
    if n_rows == 2:
        res = {0: 0.5*(M[0,0]+M[1,1]), 1: 0.5*(M[0,1]+M[1,0]),
                2: 0.5*(M[0,0]-M[1,1]), 3: 0.5j*(M[0,1]-M[1,0])}
        #print(res)
        return res
    else:
        n_qubits = int(math.log2(n_rows))
        pivr, pivc = n_rows//2, n_cols//2
        shift_x = 2*(n_qubits-1)
        shift_z = shift_x + 1

        # 1/2 (I +-Z)
        z_op = (1 << shift_z)
        i_plus_z = {0: 0.5, z_op: 0.5}
        i_minus_z = {0: 0.5, z_op: -0.5}

        # 1/2 (X +-iY)
        x_op = (1 << shift_x)
        y_op = x_op | (1 << shift_z)
        x_plus_iy = {x_op: 0.5, y_op: 0.5j}
        x_minus_iy = {x_op: 0.5, y_op: -0.5j}

        M_00 = tensor_product_pauli_dicts(recursive_mapping(M[:pivr, :pivc]), i_plus_z)
        M_11 = tensor_product_pauli_dicts(recursive_mapping(M[pivr:, pivc:]), i_minus_z)
        M_01 = tensor_product_pauli_dicts(recursive_mapping(M[:pivr, pivc:]), x_plus_iy)
        M_10 = tensor_product_pauli_dicts(recursive_mapping(M[pivr:, :pivc]), x_minus_iy)

        # Merge the 4 outputs into one additively
        d = dict()
        for k in set(M_00.keys()) | set(M_01.keys()) | set(M_10.keys()) | set(M_11.keys()):
            #d[k] = sum(Mx.get(k, 0.) for Mx in [M_00, M_01, M_10, M_11])
            d[k] = M_00.get(k, 0) + M_01.get(k, 0) + M_10.get(k, 0) + M_11.get(k, 0) # faster
        return d

#@profile
def combinatorial2(ferm_op, n_modes, n_electrons):

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f'{n=:}')

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = np.zeros((2**n, 2**n), dtype=np.complex64)
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_terms = len(ferm_op_chemist.terms)
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one
    max_int = max(ints)

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[unique_int, new_unique_int] += phase*coeff

    # Valentin: is quop_matrix sparse ? Find % of non-zero elements
    nz = np.count_nonzero(quop_matrix)
    print(f'Non-zero elements in quop_matrix = {nz} ({100*nz/4**n:5.1f}%) \t (memory :: {quop_matrix.nbytes} bytes)')
    return QubitOperator()

    # Converts matrix back into qubit operator object
    quop_ints = recursive_mapping(quop_matrix)
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            quop.terms[int_to_tuple(term, n)] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop


def combinatorial(ferm_op, n_modes, n_electrons):
    """Function to transform the fermionic Hamiltonian into a basis constructed
    in the Fock space.

    Args:
        ferm_op (FermionOperator). Fermionic operator, with alternate ordering
            as followed in the openfermion package
        n_modes (int): Number of relevant molecular orbitals, i.e. active molecular
            orbitals.
        n_electrons (int): Number of active electrons.

    Returns:
        QubitOperator: Self-explanatory.
    """

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = ceil(np.log2(n_choose_alpha * n_choose_beta))

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    qu_op = QubitOperator()
    # Check what is the effect of every term.
    for term, coeff in ferm_op_chemist.terms.items():
        # Core term.
        if not term:
            qu_op += coeff
            continue

        # Get the effect of each operator to the basis set items.
        for b, unique_int in basis_set.items():

            new_state, phase = one_body_op_on_state(term[-2:], b)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            qu_op += element_to_qubitop(n, unique_int, new_unique_int, phase*coeff)

    return qu_op


def element_to_qubitop(n_qubits, i, j, coeff=1.):
    """Map a matrix element to a qubit operator.

    Args:
        n_qubits (int): The number of qubits that the whole matrix represents.
        i (int): i row of the matrix element.
        j (int): j column of the matrix element.
        coeff (complex): Value at position i,j in the matrix.

    Returns:
        QubitOperator: Self-explanatory.
    """

    # Must add 2 to the padding because of the "0b" prefix.
    bin_i = format(i, f"#0{n_qubits+2}b")
    bin_j = format(j, f"#0{n_qubits+2}b")

    qu_op = QubitOperator("", coeff)
    for qubit, (bi, bj) in enumerate(zip(bin_i[2:][::-1], bin_j[2:][::-1])):
        if (bi, bj) == ("0", "0"):
            qu_op *= 0.5 + QubitOperator(f"Z{qubit}", 0.5)
        elif (bi, bj) == ("0", "1"):
            qu_op *= QubitOperator(f"X{qubit}", 0.5) + QubitOperator(f"Y{qubit}", 0.5j)
        elif (bi, bj) == ("1", "0"):
            qu_op *= QubitOperator(f"X{qubit}", 0.5) + QubitOperator(f"Y{qubit}", -0.5j)
        # The remaining case is 11.
        else:
            qu_op *= 0.5 + QubitOperator(f"Z{qubit}", -0.5)

    return qu_op


def basis(M, N):
    """Function to construct the combinatorial basis set, i.e. a basis set
    respecting the number of electrons and the total spin.

    Args:
        M (int): Number of spatial orbitals.
        N (int): Number of alpha or beta electrons.

    Returns:
        OrderedDict: Lexicographically sorted basis set, mapping electronic
            configuration to unique integers.
    """

    mapping = [(sigma, conf_to_integer(sigma, M)) for sigma in itertools.combinations(range(M), N)]
    return OrderedDict(mapping)


def conf_to_integer(sigma, M):
    """Function to map an electronic configuration to a unique integer, as done
    in arXiv.2205.11742 eq. (14).

    Args:
        sigma (tuple of int): Orbital indices where the electron are in the
            electronic configuration.
        M (int): Number of modes, i.e. number of spatial orbitals.

    Returns:
        int: Unique integer for the input electronic state.
    """

    # Equivalent to the number of electrons.
    N = len(sigma)

    # Eq. (14) in the reference.
    terms_k = [comb(M - sigma[N - 1 - k] - 1, k + 1) for k in range(N)]
    unique_int = comb(M, N) - 1 - np.sum(terms_k)

    return int(unique_int)


#@profile
def one_body_op_on_state(op, state_in):
    """Function to apply a a^{\dagger}_i a_j operator as described in Phys. Rev.
    A 81, 022124 (2010) eq. (8).

    Args:
        op (tuple): Operator, written as ((qubit_i, 1), (qubit_j, 0)), where 0/1
            means annihilation/creation on the specified qubit.
        state_in (tuple): Electronic configuration described as tuple of
            spinorbital indices where there is an electron.

    Returns:
        tuple: Resulting state with the same form as in the input state.
            Can be 0.
        int: Phase shift. Can be -1 or 1.
    """

    assert len(op) == 2, f"Operator {op} has length {len(op)}, but a length of 2 is expected."

    # Copy the state, then transform it into a list (it will be mutated).
    #state = deepcopy(state_in) # Not need: since state_in is unmutable, list will make a new object
    state = list(state_in)

    # Unpack the creation and annihilation operators.
    creation_op, annihilation_op = op
    creation_qubit, creation_dagger = creation_op
    annihilation_qubit, annihilation_dagger = annihilation_op

    # Confirm dagger operator to the left.
    assert creation_dagger == 1, f"The left operator in {op} is not a creation operator."
    assert annihilation_dagger == 0, f"The right operator in {op} is not an annihilation operator."

    # annihilation logics on the state.
    if annihilation_qubit in state: # use state_in.
        state.remove(annihilation_qubit)
    else:
        return (), 0

    # Creation logics on the state.
    if creation_qubit not in state:
        state.append(creation_qubit)
    else:
        return (), 0

    # Compute the phase shift.
    if annihilation_qubit > creation_qubit:
        d = sum(creation_qubit < i < annihilation_qubit for i in state)
    elif annihilation_qubit < creation_qubit:
        d = sum(annihilation_qubit < i < creation_qubit for i in state)
    else:
        d = 0

    return tuple(sorted(state)), (-1)**d
