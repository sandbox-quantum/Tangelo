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

    # Compact Hamiltonian initialization to 2^n * 2^n matrices.
    h_c = np.zeros((2**n, 2**n), dtype=complex)

    # Check what is the effect of every term.
    for term, coeff in ferm_op_chemist.terms.items():
        # Core term.
        if not term:
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
            h_c[unique_int][new_unique_int] += phase * coeff

    return h_to_qubitop(h_c, n) + ferm_op_chemist.constant


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

    mapping = [(sigma, f(sigma, M)) for sigma in itertools.combinations(range(M), N)]
    return OrderedDict(mapping)


def f(sigma, M):
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


def one_body_op_on_state(op, state_in):
    """Function to apply a a^{\dagger}_i a_j operator as described in Phys. Rev.
    A 81, 022124 (2010) eq. (8).

    Args:
        op (tuple): Operator, written as ((qubit_i, 1), (qubit_j, 0)), where 0/1
            means anhilation/creation on the specified qubit.
        state_in (tuple): Electronic configuration described as tuple of
            spinorbital indices where there is an electron.

    Returns:
        tuple or 0: Resulting state with the same form as in the input state.
            Can be 0.
        int: Phase shift. Can be -1 or 1.
    """

    assert len(op) == 2, f"Operator {op} has length {len(op)}, a length of 2 is expected."

    # Copy the state, then transform it into a list (it will be mutated).
    state = deepcopy(state_in)
    state = list(state)

    # Unpack the creation and anhilation operators.
    creation_op, anhilation_op = op
    creation_qubit, creation_dagger = creation_op
    anhilation_qubit, anhilation_dagger = anhilation_op

    # Confirm dagger operator to the left.
    assert creation_dagger == 1
    assert anhilation_dagger == 0

    # Anhilation logics on the state.
    if anhilation_qubit in state:
        state.remove(anhilation_qubit)
    else:
        return 0, 0

    # Creation logics on the state.
    if creation_qubit not in state:
        state = [*state, creation_qubit]
    else:
        return 0, 0

    # Compute the phase shift.
    if anhilation_qubit > creation_qubit:
        d = sum(creation_qubit < i < anhilation_qubit for i in state)
    elif anhilation_qubit < creation_qubit:
        d = sum(anhilation_qubit < i < creation_qubit for i in state)
    else:
        d = 0

    return tuple(sorted(state)), (-1)**d


def h_to_qubitop(h_c, n):
    """Function to map a matrix of 2^n * 2^n into a sum of tensor
    product of Pauli operators (max n elements per product, i.e. maximum number
    of qubits is n).

    Args:
        h_c (array of im): Hamiltonian matrix.

    Returns:
        QubitOperator: Self-explanatory.
    """

    qu_op = QubitOperator()

    # Go through every combinations and get the trace for the Pauli operator.
    # The trace is then taken as the coefficient for this operator.
    for pauli_tensor in itertools.product("IXYZ", repeat=n):
        pauli_word = "".join(pauli_tensor)
        term = pauli_string_to_of(pauli_word)

        term_op = QubitOperator(term, 1.)

        c_j = np.trace(h_c.conj().T @ qubit_operator_sparse(term_op, n_qubits=n).todense())
        qu_op += QubitOperator(term, c_j)

    # Normalization.
    qu_op /= np.sqrt(4**n)
    return qu_op
