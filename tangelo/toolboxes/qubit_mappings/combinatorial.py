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

"""Combinatorial mapping as described in references (1) and (2). Instead of the
occupation (or other quantity), the Fock configurations are the elements of the
basis set. In consequence, the number of required qubits scales with the number
of electronic configuration instead of the number of spinorbitals.

References:
    1. Streltsov, A. I., Alon, O. E. & Cederbaum, L. S. General mapping for
        bosonic and fermionic operators in Fock space. Phys. Rev. A 81, 022124
        (2010).
    2. Chamaki, D., Metcalf, M. & de Jong, W. A. Compact Molecular Simulation on
        Quantum Computers via Combinatorial Mapping and Variational State
        Preparation. Preprint at https://doi.org/10.48550/arXiv.2205.11742
        (2022).
"""

import itertools
from collections import OrderedDict
from math import ceil
from copy import deepcopy

import numpy as np
from scipy.special import comb
from openfermion.linalg import qubit_operator_sparse
from openfermion.transforms import chemist_ordered

from tangelo.linq.helpers.circuits.measurement_basis import pauli_string_to_of
from tangelo.toolboxes.operators import QubitOperator


def combinatorial(ferm_op, n_modes, n_electrons):
    """Function to transform the fermionic Hamiltonian into a basis constructed
    in the Fock space.

    Args:
        ferm_op (FermionOperator). Fermionic operator, with alternate ordering
            as followed in the openfermion package
        n_modes: Number of relevant molecular orbitals, i.e. active molecular
            orbitals.
        n_electrons: Number of active electrons.

    Returns:
        QubitOperator: Self-explanatory.
    """

    # The chemist ordering seperates some 1-body and 2-body terms.
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
    n = ceil(np.log2(n_choose_alpha * comb(n_modes, n_beta, exact=True)))

    # Construct the basis set where each configutation is mapped to a unique
    # integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_alpha) + int_beta
            basis_set[sigma] = unique_int

    # H_1 and H_2 initialization to 2^n * 2^n matrices.
    h_one = np.zeros((2**n, 2**n), dtype=complex)
    h_two = np.zeros((2**n, 2**n), dtype=complex)

    # Check what is the effect of every term.
    for term, coeff in ferm_op_chemist.terms.items():
        # Core term
        if not term:
            continue

        for b, unique_int in basis_set.items():
            if len(term) == 2:
                new_state, phase = one_body_op_on_state(term, b)

                if new_state:
                    new_unique_int = basis_set[new_state]
                    h_one[unique_int][new_unique_int] += phase * coeff

            elif len(term) == 4:
                new_state, phase = two_body_op_on_state(term, b)

                if new_state:
                    new_unique_int = basis_set[new_state]
                    h_two[unique_int][new_unique_int] += phase * coeff

    # Return the compact Hamiltonian H_c.
    h_c = h_one + h_two

    return h_to_qubitop(h_c, n)


def f(sigma, M):
    """TODO

    Args:

    Returns:

    """
    N = len(sigma)

    terms_k = [comb(M - sigma[N - 1 - k] - 1, k + 1) for k in range(N)]
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


def one_body_op_on_state(op, state_in):
    """"""

    assert len(op) == 2, f"{op}"

    state = deepcopy(state_in)
    state = list(state)

    creation_op, anhilation_op = op

    # Confirm dagger order to the left.
    assert creation_op[1] == 1
    assert anhilation_op[1] == 0

    creation_qubit, _ = creation_op
    anhilation_qubit, _ = anhilation_op

    if anhilation_qubit in state:
        state.remove(anhilation_qubit)
    else:
        return 0, 0.

    if creation_qubit not in state:
        state = [*state, creation_qubit]
    else:
        return 0, 0.

    if anhilation_qubit > creation_qubit:
        d = sum(creation_qubit < i < anhilation_qubit for i in state)
    elif anhilation_qubit < creation_qubit:
        d = sum(anhilation_qubit < i < creation_qubit for i in state)
    else:
        d = 0

    return tuple(sorted(state)), (-1)**d


def two_body_op_on_state(ops, state_in):
    """
    """

    op_kq = (ops[-2], ops[-1])
    state_kq, phase_kq = one_body_op_on_state(op_kq, state_in)

    if not state_kq:
        return state_kq, phase_kq

    op_sl = (ops[0], ops[1])
    state, phase_sl = one_body_op_on_state(op_sl, state_kq)

    return state, phase_kq * phase_sl


# TODO: change the algorithm to not need the full matrix.
def h_to_qubitop(h_c, n):
    """TODO

    Args:

    Returns:

    """
    qu_op = QubitOperator()

    for pauli_tensor in itertools.product("IXYZ", repeat=n):
        pauli_word = "".join(pauli_tensor)
        term = pauli_string_to_of(pauli_word)

        term_op = QubitOperator(term, 1.)

        c_j = np.trace(h_c.conj().T @ qubit_operator_sparse(term_op, n_qubits=n).todense())
        qu_op += QubitOperator(term, c_j)

    qu_op /= np.sqrt(4**n)
    return qu_op
