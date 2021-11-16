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

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Module to create and manipulate unitary coupled cluster operators for
   open shell systems. This requires the number of alpha and beta electrons
   to be specified.
"""

import itertools

from openfermion.utils import down_index, up_index

from tangelo.toolboxes.operators import FermionOperator


def uccsd_openshell_paramsize(n_spinorbitals, n_alpha_electrons, n_beta_electrons):
    """Determine number of independent amplitudes for open-shell UCCSD
     Args:
        n_spinorbitals(int): Number of spin-orbitals in the system
        n_alpha_electrons(int): Number of alpha electrons in the reference state
        n_beta_electrons(int): Number of beta electrons in the reference state
    Returns:
        The number of unique single amplitudes, double amplitudes
        and the number of single alpha and beta amplitudes, as well as the
        number of double alpha-alpha, beta-beta and alpha-beta amplitudes
    """
    if n_spinorbitals % 2 != 0:
        raise ValueError("The total number of spin-orbitals should be even.")

    # Compute the number of occupied and virtual alpha and beta orbitals
    n_orb_a_b = n_spinorbitals // 2
    n_occ_a = n_alpha_electrons
    n_occ_b = n_beta_electrons
    n_virt_a = n_orb_a_b - n_alpha_electrons
    n_virt_b = n_orb_a_b - n_beta_electrons

    # Calculate the number of alpha single amplitudes
    n_single_a = n_occ_a * n_virt_a

    # Calculate the number of beta single amplitudes
    n_single_b = n_occ_b * n_virt_b

    # Calculate the total number of single amplitudes
    n_single_amplitudes = n_single_a + n_single_b

    # Calculate the number of alpha-alpha double amplitudes
    n_double_aa = n_occ_a * (n_occ_a - 1) * n_virt_a * (n_virt_a - 1) // 4

    # Calculate the number of beta-beta double amplitudes
    n_double_bb = n_occ_b * (n_occ_b - 1) * n_virt_b * (n_virt_b - 1) // 4

    # Calculate the number of alpha-beta double amplitudes
    n_double_ab = n_occ_a * n_occ_b * n_virt_a * n_virt_b

    # Calculate the total number of double amplitudes
    n_double_amplitudes = n_double_aa + n_double_bb + n_double_ab

    return n_single_amplitudes, n_double_amplitudes, n_single_a, n_single_b,\
        n_double_aa, n_double_bb, n_double_ab


def uccsd_openshell_generator(packed_amplitudes, n_spinorbitals, n_alpha_electrons,
                              n_beta_electrons, anti_hermitian=True):
    r"""Create an open-shell UCCSD generator for a system with n_alpha_electrons and
       n_beta_electrons
    This function generates a FermionOperator for a UCCSD generator designed
        to act on a single reference state consisting of n_spinorbitals spin orbitals
        and n_alpha_electrons alpha electrons and n_beta_electrons beta electrons,
        that is a high-spin open-shell operator
    Args:
        packed_amplitudes(list): List storing the unique single
            and double excitation amplitudes for an open-shell UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_spinorbitals(int): Number of spin-orbitals used to represent the system
        n_alpha_electrons(int): Number of alpha electrons in the physical system.
        n_beta_electrons(int): Number of beta electrons in the physical system.
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing
    Returns:
        generator(FermionOperator): Generator of the UCCSD operator that
            builds the open-shell UCCSD wavefunction.
    """
    if n_spinorbitals % 2 != 0:
        raise ValueError("The total number of spin-orbitals should be even.")

    # Compute the number of occupied and virtual alpha and beta orbitals
    n_orb_a_b = n_spinorbitals // 2
    n_occ_a = n_alpha_electrons
    n_occ_b = n_beta_electrons
    n_virt_a = n_orb_a_b - n_alpha_electrons
    n_virt_b = n_orb_a_b - n_beta_electrons

    # Unpack the single and double amplitudes
    _, _, n_single_a, n_single_b, \
        n_double_aa, n_double_bb, _ = uccsd_openshell_paramsize(n_spinorbitals, n_alpha_electrons, n_beta_electrons)

    # Define the various increments for the sizes of the orbital spaces
    n_s_1 = n_single_a
    n_s_2 = n_single_a + n_single_b
    n_d_1 = n_s_2 + n_double_aa
    n_d_2 = n_d_1 + n_double_bb

    # Alpha single amplitudes
    t1_a = packed_amplitudes[:n_s_1]
    # Beta single amplitudes
    t1_b = packed_amplitudes[n_s_1:n_s_2]
    # Alpha-Alpha double amplitudes
    t2_aa = packed_amplitudes[n_s_2:n_d_1]
    # Beta-Beta double amplitudes
    t2_bb = packed_amplitudes[n_d_1:n_d_2]
    # Alpha-Beta double amplitudes
    t2_ab = packed_amplitudes[n_d_2:]

    # Initialize operator
    generator = FermionOperator()

    # Generate all spin-conserving single excitations
    # for the alpha spin case
    for i, (p, q) in enumerate(
            itertools.product(range(n_virt_a), range(n_occ_a))):

        # Get indices of spatial orbitals
        virtual_a = n_occ_a + p
        occupied_a = q

        # Map the alpha index to the proper spin orbital index
        virtual_so = up_index(virtual_a)
        occupied_so = up_index(occupied_a)

        # Generate the alpha single excitations
        coeff = t1_a[i]
        generator += FermionOperator((
            (virtual_so, 1),
            (occupied_so, 0)),
            coeff)
        if anti_hermitian:
            generator += FermionOperator((
                (occupied_so, 1),
                (virtual_so, 0)),
                -coeff)

    # Generate all spin-conserving single excitations
    # for the beta spin case
    for i, (p, q) in enumerate(
            itertools.product(range(n_virt_b), range(n_occ_b))):

        # Get indices of spatial orbitals
        virtual_b = n_occ_b + p
        occupied_b = q

        # Map the beta index to the proper spin orbital index
        virtual_so = down_index(virtual_b)
        occupied_so = down_index(occupied_b)

        # Generate the beta single excitations
        coeff = t1_b[i]
        generator += FermionOperator((
            (virtual_so, 1),
            (occupied_so, 0)),
            coeff)
        if anti_hermitian:
            generator += FermionOperator((
                (occupied_so, 1),
                (virtual_so, 0)),
                -coeff)

    # Generate all spin-conserving alpha-alpha double excitations
    for i, ((s, q), (r, p)) in enumerate(
            itertools.product(
                itertools.combinations(range(n_occ_a), 2), itertools.combinations(range(n_virt_a), 2)
                )):

        # Get indices of orbitals
        virtual_1 = n_occ_a + p
        occupied_1 = q
        virtual_2 = n_occ_a + r
        occupied_2 = s

        # Generate double excitations
        coeff = t2_aa[i]

        # Map the occupied and virtual alpha indices to spin-orbital indices
        virtual_1_a = up_index(virtual_1)
        occupied_1_a = up_index(occupied_1)
        virtual_2_a = up_index(virtual_2)
        occupied_2_a = up_index(occupied_2)

        generator += FermionOperator((
            (virtual_1_a, 1),
            (occupied_1_a, 0),
            (virtual_2_a, 1),
            (occupied_2_a, 0)),
            coeff)
        if anti_hermitian:
            generator += FermionOperator((
                (occupied_2_a, 1),
                (virtual_2_a, 0),
                (occupied_1_a, 1),
                (virtual_1_a, 0)),
                -coeff)
    # Generate all spin-conserving beta-beta double excitations
    for i, ((s, q), (r, p)) in enumerate(
            itertools.product(
                itertools.combinations(range(n_occ_b), 2), itertools.combinations(range(n_virt_b), 2)
            )):
        # Get indices of orbitals
        virtual_1 = n_occ_b + p
        occupied_1 = q
        virtual_2 = n_occ_b + r
        occupied_2 = s

        # Generate double excitations
        coeff = t2_bb[i]

        # Map the occupied and virtual alpha indices to spin-orbital indices
        virtual_1_b = down_index(virtual_1)
        occupied_1_b = down_index(occupied_1)
        virtual_2_b = down_index(virtual_2)
        occupied_2_b = down_index(occupied_2)

        generator += FermionOperator((
            (virtual_1_b, 1),
            (occupied_1_b, 0),
            (virtual_2_b, 1),
            (occupied_2_b, 0)),
            coeff)
        if anti_hermitian:
            generator += FermionOperator((
                (occupied_2_b, 1),
                (virtual_2_b, 0),
                (occupied_1_b, 1),
                (virtual_1_b, 0)),
                -coeff)
        # Generate all spin-conserving alpha-beta double excitations
    for i, (p, q, r, s) in enumerate(
            itertools.product(
                range(n_virt_a), range(n_occ_a), range(n_virt_b), range(n_occ_b))):

        # Get indices of orbitals
        virtual_1 = n_occ_a + p
        occupied_1 = q
        virtual_2 = n_occ_b + r
        occupied_2 = s

        # Generate double excitations
        coeff = t2_ab[i]

        # Map the alpha and beta occupied and virtual orbitals to spinorbitals
        virtual_1_a = up_index(virtual_1)
        occupied_1_a = up_index(occupied_1)
        virtual_2_b = down_index(virtual_2)
        occupied_2_b = down_index(occupied_2)

        generator += FermionOperator((
            (virtual_1_a, 1),
            (occupied_1_a, 0),
            (virtual_2_b, 1),
            (occupied_2_b, 0)),
            coeff)
        if anti_hermitian:
            generator += FermionOperator((
                (occupied_2_b, 1),
                (virtual_2_b, 0),
                (occupied_1_a, 1),
                (virtual_1_a, 0)),
                -coeff)
    return generator


def uccsd_openshell_get_packed_amplitudes(alpha_double_amplitudes, beta_double_amplitudes,
                                          alpha_beta_double_amplitudes, n_spinorbitals, n_alpha_electrons,
                                          n_beta_electrons, alpha_single_amplitudes=None,
                                          beta_single_amplitudes=None):
    r"""Convert amplitudes for use with the open-shell UCCSD (e.g. from a UHF MP2 guess)
    The output list contains only the non-redundant amplitudes that are
    relevant to open-shell UCCSD, in an order suitable for use with the function
    `uccsd_openshell_generator`.
    Args:
        alpha_single_amplitudes(ndarray): [N_virtual_alpha x N_occupied_alpha]
            array string the alpha single excitation amplitudes corresponding
            to t[i_alpha,a_alpha] * (a_a_alpha^\dagger a_i_alpha)
        beta_single_amplitudes(ndarray): [N_virtual_beta x N_occupied_beta]
            array string the beta single excitation amplitudes corresponding
            to t[i_beta,a_beta] * (a_a_beta^\dagger a_i_beta)
        alpha_double_amplitudes(ndarray): [N_occupied_alpha x N_occupied_alpha
            x N_virtual_alpha x N_virtual_alpha] array storing the alpha-alpha
            double excitation amplitudes corresponding to
            t[i_alpha,j_alpha,a_alpha,b_alpha] * (a_a_alpha^\dagger a_i_alpha
            a_b_alpha^\dagger a_j_alpha - H.C.)
        beta_double_amplitudes(ndarray): [N_occupied_beta x N_occupied_beta
            x N_virtual_beta x N_virtual_beta] array storing the beta-beta
            double excitation amplitudes corresponding to
            t[i_beta,j_beta,a_beta,b_beta] * (a_a_beta^\dagger a_i_beta
            a_b_beta^\dagger a_j_beta - H.C.)
        alpha_beta_double_amplitudes(ndarray): [N_occupied_alpha x N_occupied_beta
            x N_virtual_alpha x N_virtual_beta] array storing the alpha-beta
            double excitation amplitudes corresponding to
            t[i_alpha,j_beta,a_alpha,b_beta] * (a_a_alpha^\dagger a_i_alpha
            a_b_beta^\dagger a_j_beta - H.C.)
        n_spinorbitals(int): Number of spin-orbitals used to represent the system
        n_alpha_electrons(int): Number of alpha electrons in the physical system.
        n_beta_electrons(int): Number of beta electrons in the physical system
        alpha_single_amplitudes(ndarray optional): optional [N_occupied_alpha
            x N_virtual_alpha] array string the alpha single excitation
            amplitudes corresponding to t[i_alpha,a_alpha]
            * (a_a_alpha^\dagger a_i_alpha)
        beta_single_amplitudes(ndarray optional): optional [N_occupied_beta
            x N_virtual_beta] array string the beta single excitation
            amplitudes corresponding to t[i_beta,a_beta]
            * (a_a_beta^\dagger a_i_beta)
    Returns:
        packed_amplitudes(list): List storing the unique single (zero in UHF-MP2)
            and double excitation amplitudes for an open-shell UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
    """

    if n_spinorbitals % 2 != 0:
        raise ValueError("The total number of spin-orbitals should be even.")

    # Compute the number of occupied and virtual alpha and beta orbitals
    n_orb_a_b = n_spinorbitals // 2
    n_occ_a = n_alpha_electrons
    n_occ_b = n_beta_electrons
    n_virt_a = n_orb_a_b - n_alpha_electrons
    n_virt_b = n_orb_a_b - n_beta_electrons

    # Calculate the number of non-redundant single and double amplitudes
    _, _, n_single_a, n_single_b, _, _, _ = uccsd_openshell_paramsize(n_spinorbitals, n_alpha_electrons, n_beta_electrons)

    # packed amplitudes list
    packed_amplitudes = []

    #  Extract all the single excitation amplitudes
    #  for the alpha spin case (they are zero for a UHF
    #  reference due to the Brillouin condition)
    if alpha_single_amplitudes is None:
        packed_amplitudes = [0.]*n_single_a
    else:
        for (p, q) in itertools.product(range(n_virt_a), range(n_occ_a)):

            # Get the amplitude
            packed_amplitudes.append(alpha_single_amplitudes[q, p])

    #  Extract all the single excitation amplitudes
    #  for the beta spin case (they are zero for a UHF
    #  reference due to the Brillouin condition)
    if beta_single_amplitudes is None:
        packed_amplitudes += [0.]*n_single_b
    else:
        for (p, q) in itertools.product(range(n_virt_b), range(n_occ_b)):

            # Get the amplitude
            packed_amplitudes.append(beta_single_amplitudes[q, p])

    # Extract all of the non-redundant alpha-alpha double
    # excitation amplitudes
    for ((s, q), (r, p)) in itertools.product(
                itertools.combinations(range(n_occ_a), 2), itertools.combinations(range(n_virt_a), 2)
                ):

        # Get the amplitude
        packed_amplitudes.append(alpha_double_amplitudes[q, s, p, r])

    # Extract all of the non-redundant beta-beta double
    # excitation amplitudes
    for ((s, q), (r, p)) in itertools.product(
                itertools.combinations(range(n_occ_b), 2), itertools.combinations(range(n_virt_b), 2)
                ):

        # Get the amplitude
        packed_amplitudes.append(beta_double_amplitudes[q, s, p, r])

    # Extract all of the non-redundant alpha-beta double
    # excitation amplitudes
    for (p, q, r, s) in itertools.product(
                range(n_virt_a), range(n_occ_a), range(n_virt_b), range(n_occ_b)):

        # Get the amplitude
        packed_amplitudes.append(alpha_beta_double_amplitudes[q, s, p, r])

    return packed_amplitudes
