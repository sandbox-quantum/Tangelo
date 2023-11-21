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

import itertools
import math
from collections import OrderedDict

import numpy as np
from scipy.special import comb
from openfermion.transforms import chemist_ordered

from tangelo.toolboxes.operators import QubitOperator


ZERO_TOLERANCE = 1e-8


def int_to_tuple(integer, n_qubits):
    """ Convert a qubit Hamiltonian term in integer encoding (stabilizer representation) into
    an Openfermion-style tuple.

    Bits in the binary representation of the integer encode the Pauli operators that need to be applied
    to each qubit. Each consecutive pair of bits encode information for a specific qubit.

    Args:
        integer (int): integer to decode. Its binary representation has 2*n_qubits bits
        n_qubits (int): number of qubits in the term

    Returns:
        tuple: OpenFermion-style term including up to n_qubits qubits
    """

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
    """ Perform the tensor product of 2 Pauli operators by using their underlying dictionary of terms.

    Args:
        pa (dict[int -> complex]): first Pauli dictionary
        pb (dict[int -> complex]): second Pauli dictionary

    Returns:
        dict[int -> complex]: tensor product of Pauli operators
    """
    pp = dict()
    for ta, ca in pa.items():
        for tb, cb in pb.items():
            pp[ta ^ tb] = ca*cb
    return pp


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

    # Copy the state, then transform it into a list (it will be mutated).
    state = list(state_in)

    # Unpack the creation and annihilation operators.
    creation_op, annihilation_op = op
    creation_qubit, creation_dagger = creation_op
    annihilation_qubit, annihilation_dagger = annihilation_op

    # annihilation logics on the state.
    if annihilation_qubit in state:
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


def recursive_mapping(M):
    """ Maps an arbitrary square matrix representing an operator via Pauli decomposition.

    Args:
        M (np.array(np, complex, np.complex)): Square matrix representing the operator.

    Returns:
        dict[int -> complex]: Pauli operator encoded as a dictionary
    """

    n_rows, n_cols = M.shape

    # Bottom of recursion: 2x2 matrix case
    if n_rows == 2:
        res = {0: 0.5*(M[0,0]+M[1,1]), 1: 0.5*(M[0,1]+M[1,0]), 2: 0.5*(M[0,0]-M[1,1]), 3: 0.5j*(M[0,1]-M[1,0])}
        return res
    else:
        n_qubits = int(math.log2(n_rows))
        piv = n_rows//2
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

        M_00 = tensor_product_pauli_dicts(recursive_mapping(M[:piv, :piv]), i_plus_z)
        M_11 = tensor_product_pauli_dicts(recursive_mapping(M[piv:, piv:]), i_minus_z)
        M_01 = tensor_product_pauli_dicts(recursive_mapping(M[:piv, piv:]), x_plus_iy)
        M_10 = tensor_product_pauli_dicts(recursive_mapping(M[piv:, :piv]), x_minus_iy)

        # Merge the 4 outputs into one additively
        for d in M_01, M_10, M_11:
            for (k, v) in d.items():
                M_00[k] = M_00.get(k, 0.) + v
        return M_00


def combinatorial(ferm_op, n_modes, n_electrons):
    """ Function to transform the fermionic Hamiltonian into a basis constructed
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
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in OpenFermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = np.zeros((2**n, 2**n), dtype=np.complex64)
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items()))

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

    # Convert matrix back into qubit operator object
    quop_ints = recursive_mapping(quop_matrix)
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            quop.terms[int_to_tuple(term, n)] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop


def basis(M, N):
    """ Function to construct the combinatorial basis set, i.e. a basis set
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
    """ Function to map an electronic configuration to a unique integer, as done
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
