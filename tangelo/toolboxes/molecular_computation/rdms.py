# Copyright 2023 1QB Information Technologies Inc.
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

"""Module containing functions to manipulate 1- and 2-RDMs."""

import itertools as it

import numpy as np

from tangelo.toolboxes.molecular_computation.coefficients import spatial_from_spinorb
from tangelo.linq.helpers import pauli_string_to_of, pauli_of_to_string, get_compatible_bases
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.toolboxes.post_processing import Histogram, aggregate_histograms
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping, get_qubit_number


def matricize_2rdm(two_rdm, n_orbitals):
    """Turns the two_rdm tensor into a matrix for test purposes."""

    l = 0
    sq = n_orbitals * n_orbitals
    jpqrs = np.zeros((n_orbitals, n_orbitals), dtype=int)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            jpqrs[i, j] = l
            l += 1

    rho = np.zeros((sq, sq), dtype=complex)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            ij = jpqrs[i, j]
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    kl = jpqrs[k, l]
                    rho[ij, kl] += two_rdm[i, k, j, l]
    return rho


def compute_rdms(ferm_ham, mapping, up_then_down, exp_vals=None, exp_data=None, shadow=None, return_spinsum=True, **eval_args):
    """
    Compute the 1- and 2-RDM and their spin-summed versions
    using a FermionOperator and experimental frequency data in the form of a
    classical shadow or a dictionary of frequency histograms.
    Exactly one of the following must be provided by the user:
    exp_vals, exp_data or shadow

    Args:
        ferm_ham (FermionicOperator): Fermionic operator with n_spinorbitals, n_electrons, and spin defined
        mapping (str): Qubit mapping
        up_then_down (bool): Spin ordering for the mapping
        exp_vals (dict): Optional, dictionary of Pauli word expectation values
        exp_data (dict): Optional, dictionary of {basis: histogram} where basis is the measurement basis
            and histogram is a {bitstring: frequency} dictionary
        shadow (ClassicalShadow): Optional, a classical shadow object
        return_spinsum (bool): Optional, if True, return also the spin-summed RDMs
        eval_args: Optional arguments to pass to the ClassicalShadow object

    Returns:
        complex array: 1-RDM
        complex array: 2-RDM
        complex array: spin-summed 1-RDM
        complex array: spin-summed 2-RDM
    """
    onerdm = np.zeros((ferm_ham.n_spinorbitals,) * 2, dtype=complex)
    twordm = np.zeros((ferm_ham.n_spinorbitals,) * 4, dtype=complex)

    # Optional arguments are mutually exclusive, return error if several of them have been passed
    if [exp_vals, exp_data, shadow].count(None) != 2:
        raise RuntimeError("Arguments exp_vals, exp_data and shadow are mutually exclusive. Provide exactly one of them.")

    # Initialize exp_vals
    if isinstance(exp_vals, dict) and set(map(type, exp_vals)) == {str}:
        exp_vals = {pauli_string_to_of(term): exp_val for term, exp_val in exp_vals.items()}

    if isinstance(exp_data, dict) and set(map(type, exp_data)) == {str}:
        exp_data = {pauli_string_to_of(term): data for term, data in exp_data.items()}

    if exp_vals is None:
        exp_vals = dict()

    n_qubits = get_qubit_number(mapping, ferm_ham.n_spinorbitals)

    # Go over all terms in fermionic Hamiltonian
    for term in ferm_ham.terms:
        length = len(term)

        if not term:
            continue

        # Fermionic term with a prefactor of 1.0.
        fermionic_term = FermionOperator(term, 1.0)

        qubit_term = fermion_to_qubit_mapping(fermion_operator=fermionic_term,
                                              n_spinorbitals=ferm_ham.n_spinorbitals,
                                              n_electrons=ferm_ham.n_electrons,
                                              mapping=mapping,
                                              up_then_down=up_then_down,
                                              spin=ferm_ham.spin)
        qubit_term.compress()

        # Loop to go through all qubit terms.
        eigenvalue = 0.

        if isinstance(shadow, ClassicalShadow):
            eigenvalue = shadow.get_observable(qubit_term, **eval_args)
        else:
            for qterm, coeff in qubit_term.terms.items():
                if coeff.real != 0:

                    # qterm = (), i.e. tensor product of I.
                    if not qterm:
                        exp_val = 1.
                    # Already computed expectation value of qterm.
                    elif qterm in exp_vals:
                        exp_val = exp_vals[qterm]
                    # Case where there is at least one missing entry in exp_vals
                    # (in this case, exp_data is None and we cannot compute the
                    # eigenvalue).
                    elif exp_data is None:
                        raise RuntimeError(f"Missing {qterm} entry in exp_vals.")
                    # Expectation value that can be computed from exp_data.
                    else:
                        ps = pauli_of_to_string(qterm, n_qubits)
                        bases = get_compatible_bases(ps, [pauli_of_to_string(term, n_qubits) for term in exp_data.keys()])

                        if not bases:
                            raise RuntimeError(f"No experimental data for basis {qterm}.")

                        hist = aggregate_histograms(*[Histogram(exp_data[pauli_string_to_of(basis)]) for basis in bases])
                        exp_val = hist.get_expectation_value(qterm, 1.)
                        exp_vals[qterm] = exp_val

                    eigenvalue += coeff * exp_val

        # Put the values in np arrays (differentiate 1- and 2-RDM)
        if length == 2:
            iele, jele = (int(ele[0]) for ele in tuple(term[0:2]))
            onerdm[iele, jele] += eigenvalue
        elif length == 4:
            iele, jele, kele, lele = (int(ele[0]) for ele in tuple(term[0:4]))
            twordm[iele, lele, jele, kele] += eigenvalue

    if return_spinsum:
        onerdm_spinsum = np.zeros((ferm_ham.n_spinorbitals//2,) * 2, dtype=complex)
        twordm_spinsum = np.zeros((ferm_ham.n_spinorbitals//2,) * 4, dtype=complex)

        # Construct spin-summed 1-RDM.
        for i, j in it.product(range(onerdm.shape[0]), repeat=2):
            onerdm_spinsum[i//2, j//2] += onerdm[i, j]

        # Construct spin-summed 2-RDM.
        for i, j, k, l in it.product(range(twordm.shape[0]), repeat=4):
            twordm_spinsum[i//2, j//2, k//2, l//2] += twordm[i, j, k, l]

        return onerdm, twordm, onerdm_spinsum, twordm_spinsum
    else:
        return onerdm, twordm


def energy_from_rdms(ferm_op, one_rdm, two_rdm):
    """Computes the molecular energy from one- and two-particle reduced
    density matrices (RDMs). Coefficients (integrals) are computed from the
    fermionic Hamiltonian provided.

    Args:
        ferm_op (FermionOperator): Self-explanatory.
        one_rdm (numpy.array): One-particle density matrix in MO basis.
        two_rdm (numpy.array): Two-particle density matrix in MO basis.

    Returns:
        float: Molecular energy.
    """

    core_constant, one_electron_coeffs, two_electron_coeffs = ferm_op.get_coeffs()
    one_electron_integrals, two_electron_integrals = spatial_from_spinorb(one_electron_coeffs, two_electron_coeffs)

    # PQRS convention in openfermion:
    # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
    # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
    # The convention is not the same with PySCF integrals. So, a change is
    # reverse back after performing the truncation for frozen orbitals
    two_electron_integrals = two_electron_integrals.transpose(0, 3, 1, 2)

    # Computing the total energy from integrals and provided RDMs.
    e = core_constant + np.sum(one_electron_integrals * one_rdm) + np.sum(two_electron_integrals * two_rdm)

    return e.real


def pad_rdms_with_frozen_orbitals_restricted(sec_mol, onerdm, twordm):
    """Function to pad the RDMs with the frozen orbitals data. It is based on
    the pyscf.cccsd_rdm code, where we can set with_frozen=True.

    Source:
        https://github.com/pyscf/pyscf/blob/master/pyscf/cc/ccsd_rdm.py

    Args:
        sec_mol (SecondQuantizedMolecule): Self-explanatory.
        onerdm (numpy.array): One-particle reduced density matrix (shape of
            (N_active_mos,)*2).
        twordm (numpy.array): Two-particle reduced density matrix (shape of
            (N_active_mos,)*4).

    Returns:
        numpy.array: One-particle reduced density matrix (shape of
            (N_total_mos,)*2).
        numpy.array: Two-particle reduced density matrix (shape of
            (N_total_mos,)*4).
    """
    from pyscf.lib import takebak_2d

    if sec_mol.uhf:
        raise NotImplementedError("The RDMs padding with an UHF mean-field is not implemented.")

    # Defining the number of MOs and occupation numbers with and without the
    # frozen orbitals.
    n_mos = sec_mol.n_mos
    n_mos0 = sec_mol.n_active_mos
    n_occ = np.count_nonzero(sec_mol.mo_occ > 0)
    n_occ0 = n_occ - len(sec_mol.frozen_occupied)
    moidx = np.array(sec_mol.active_mos)

    # Creating a dummy one rdm with all diagonal elements set to 2. After that,
    # the true one rdm is embedded in this bigger matrix.
    onerdm_padded = np.zeros((n_mos,)*2, dtype=onerdm.dtype)
    onerdm_padded[np.diag_indices(n_occ)] = 2.
    onerdm_padded[moidx[:, None], moidx] = onerdm

    # Deleting the one rdm contribution in the two rdm. This must be done to
    # redo the operation later with the one rdm with the frozen orbital.
    twordm = twordm.transpose(1, 0, 3, 2)

    onerdm_without_diag = np.copy(onerdm)
    onerdm_without_diag[np.diag_indices(n_occ0)] -= 2

    onerdm_without_diag_times_2 = onerdm_without_diag * 2
    onerdm_without_diag_T = onerdm_without_diag.T

    for i in range(n_occ0):
        twordm[i, i, :, :] -= onerdm_without_diag_times_2
        twordm[:, :, i, i] -= onerdm_without_diag_times_2
        twordm[:, i, i, :] += onerdm_without_diag
        twordm[i, :, :, i] += onerdm_without_diag_T

    for i, j in it.product(range(n_occ0), repeat=2):
        twordm[i, i, j, j] -= 4
        twordm[i, j, j, i] += 2

    # Creating a dummy two rdm.
    dm2 = np.zeros((n_mos,)*4, dtype=twordm.dtype)

    idx = (moidx.reshape(-1, 1) * n_mos + moidx).ravel()

    # This part is meant to replicate
    # https://github.com/pyscf/pyscf/blob/8b3fef8cf18f10d430261d4a8bea21fadf19bb1f/pyscf/cc/ccsd_rdm.py#L343-L351.
    # The output is not catched, maybe for memory efficiency purposes (elements
    # of dm2 changed inplace?).
    takebak_2d(dm2.reshape(n_mos**2, n_mos**2),
               twordm.reshape(n_mos0**2, n_mos0**2), idx, idx)
    twordm_padded = dm2

    # The next few lines will reembed the on rdm, but with the frozen orbital
    # elements of the one rdm matrix.
    onerdm_padded_without_diag = np.copy(onerdm_padded)
    onerdm_padded_without_diag[np.diag_indices(n_occ)] -= 2

    onerdm_padded_without_diag_times_2 = onerdm_padded_without_diag * 2
    onerdm_padded_without_diag_T = onerdm_padded_without_diag.T

    for i in range(n_occ):
        twordm_padded[i, i, :, :] += onerdm_padded_without_diag_times_2
        twordm_padded[:, :, i, i] += onerdm_padded_without_diag_times_2
        twordm_padded[:, i, i, :] -= onerdm_padded_without_diag
        twordm_padded[i, :, :, i] -= onerdm_padded_without_diag_T

    for i, j in it.product(range(n_occ), repeat=2):
        twordm_padded[i, i, j, j] += 4
        twordm_padded[i, j, j, i] -= 2

    twordm_padded = twordm_padded.transpose(1, 0, 3, 2)

    return onerdm_padded, twordm_padded


def pad_rdms_with_frozen_orbitals_unrestricted(sec_mol, onerdm, twordm):
    """Function to pad the RDMs with the frozen orbitals data. It is based on
    the pyscf.ucccsd_rdm code, where we can set with_frozen=True.

    Source:
        https://github.com/pyscf/pyscf/blob/master/pyscf/cc/uccsd_rdm.py

    Args:
        sec_mol (SecondQuantizedMolecule): Self-explanatory.
        onerdm (tuple of arrays): Tuple of alpha and beta one-particle
            reduced density matrix (shape of (N_active_mos,)*2).
        twordm (tuple of arrays): Tuple of alpha-alpha, alpha-beta and beta-beta
            two-particle reduced density matrix (shape of (N_active_mos,)*4).

    Returns:
        tuple of arrays: Tuple of alpha and beta one-particle reduced density
            matrix (shape of (N_total_mos,)*2).
        tuple of arrays: Tuple of alpha-alpha, alpha-beta and beta-beta
            two-particle reduced density matrix (shape of (N_total_mos,)*4).
    """
    from pyscf.lib import takebak_2d

    # Unpacking the tuples.
    onerdm_a, onerdm_b = onerdm
    twordm_aa, twordm_ab, twordm_bb = twordm

    # Defining the number of MOs and occupation numbers with and without the
    # frozen orbitals.
    n_mos_a, n_mos_b = sec_mol.mo_occ[0].size, sec_mol.mo_occ[1].size
    n_mos0_a, n_mos0_b = sec_mol.n_active_mos
    n_occ_a = np.count_nonzero(sec_mol.mo_occ[0] > 0)
    n_occ_b = np.count_nonzero(sec_mol.mo_occ[1] > 0)
    n_occ0_a = n_occ_a - len(sec_mol.frozen_occupied[0])
    n_occ0_b = n_occ_b - len(sec_mol.frozen_occupied[1])

    moidx_a = np.array(sec_mol.active_mos[0])
    moidx_b = np.array(sec_mol.active_mos[1])

    # Creating a dummy one rdm with all diagonal elements set to 1. After that,
    # the true one rdm is embedded in this bigger matrix.
    onerdm_a_padded = np.zeros((n_mos_a,)*2, dtype=onerdm_a.dtype)
    onerdm_b_padded = np.zeros((n_mos_b,)*2, dtype=onerdm_b.dtype)

    onerdm_a_padded[np.diag_indices(n_occ_a)] = 1.
    onerdm_b_padded[np.diag_indices(n_occ_b)] = 1.

    onerdm_a_padded[moidx_a[:, None], moidx_a] = onerdm_a
    onerdm_b_padded[moidx_b[:, None], moidx_b] = onerdm_b

    # Deleting the one rdm contribution in the two rdm. This must be done to
    # redo the operation later with the one rdm with the frozen orbital.
    twordm_aa = twordm_aa.transpose(1, 0, 3, 2)
    twordm_bb = twordm_bb.transpose(1, 0, 3, 2)
    twordm_ab = twordm_ab.transpose(1, 0, 3, 2)

    onerdm_without_diag_a = np.copy(onerdm_a)
    onerdm_without_diag_a[np.diag_indices(n_occ0_a)] -= 1
    onerdm_without_diag_b = np.copy(onerdm_b)
    onerdm_without_diag_b[np.diag_indices(n_occ0_b)] -= 1

    onerdm_without_diag_a_T = onerdm_without_diag_a.T
    onerdm_without_diag_b_T = onerdm_without_diag_b.T

    for i in range(n_occ0_a):
        twordm_aa[i, i, :, :] -= onerdm_without_diag_a
        twordm_aa[:, :, i, i] -= onerdm_without_diag_a
        twordm_aa[:, i, i, :] += onerdm_without_diag_a
        twordm_aa[i, :, :, i] += onerdm_without_diag_a_T
        twordm_ab[i, i, :, :] -= onerdm_without_diag_b

    for i in range(n_occ0_b):
        twordm_bb[i, i, :, :] -= onerdm_without_diag_b
        twordm_bb[:, :, i, i] -= onerdm_without_diag_b
        twordm_bb[:, i, i, :] += onerdm_without_diag_b
        twordm_bb[i, :, :, i] += onerdm_without_diag_b_T
        twordm_ab[:, :, i, i] -= onerdm_without_diag_a

    for i, j in it.product(range(n_occ0_a), repeat=2):
        twordm_aa[i, i, j, j] -= 1
        twordm_aa[i, j, j, i] += 1

    for i, j in it.product(range(n_occ0_b), repeat=2):
        twordm_bb[i, i, j, j] -= 1
        twordm_bb[i, j, j, i] += 1

    for i, j in it.product(range(n_occ0_a), range(n_occ0_b), repeat=1):
        twordm_ab[i, i, j, j] -= 1

    # Creating a dummy two rdm.
    dm2_aa = np.zeros((n_mos_a,)*4, dtype=twordm_aa.dtype)
    dm2_bb = np.zeros((n_mos_b,)*4, dtype=twordm_bb.dtype)
    dm2_ab = np.zeros((n_mos_a, n_mos_a, n_mos_b, n_mos_b), dtype=twordm_ab.dtype)

    idx_a = (moidx_a.reshape(-1, 1) * n_mos_a + moidx_a).ravel()
    idx_b = (moidx_b.reshape(-1, 1) * n_mos_b + moidx_b).ravel()

    # This part is meant to replicate
    # https://github.com/pyscf/pyscf/blob/8b3fef8cf18f10d430261d4a8bea21fadf19bb1f/pyscf/cc/uccsd_rdm.py#L561-L583.
    # The output is not catched, maybe for memory efficiency purposes (elements
    # of dm2 changed inplace?).
    takebak_2d(dm2_aa.reshape(n_mos_a**2, n_mos_a**2),
               twordm_aa.reshape(n_mos0_a**2, n_mos0_a**2), idx_a, idx_a)
    takebak_2d(dm2_bb.reshape(n_mos_b**2, n_mos_b**2),
               twordm_bb.reshape(n_mos0_b**2, n_mos0_b**2), idx_b, idx_b)
    takebak_2d(dm2_ab.reshape(n_mos_a**2, n_mos_b**2),
               twordm_ab.reshape(n_mos0_a**2, n_mos0_b**2), idx_a, idx_b)
    twordm_aa_padded, twordm_ab_padded, twordm_bb_padded = dm2_aa, dm2_ab, dm2_bb

    # The next few lines will reembed the on rdm, but with the frozen orbital
    # elements of the one rdm matrix.
    onerdm_a_padded_without_diag = np.copy(onerdm_a_padded)
    onerdm_a_padded_without_diag[np.diag_indices(n_occ_a)] -= 1

    onerdm_b_padded_without_diag = np.copy(onerdm_b_padded)
    onerdm_b_padded_without_diag[np.diag_indices(n_occ_b)] -= 1

    onerdm_a_padded_without_diag_T = onerdm_a_padded_without_diag.T
    onerdm_b_padded_without_diag_T = onerdm_b_padded_without_diag.T

    for i in range(n_occ_a):
        twordm_aa_padded[i, i, :, :] += onerdm_a_padded_without_diag
        twordm_aa_padded[:, :, i, i] += onerdm_a_padded_without_diag
        twordm_aa_padded[:, i, i, :] -= onerdm_a_padded_without_diag
        twordm_aa_padded[i, :, :, i] -= onerdm_a_padded_without_diag_T
        twordm_ab_padded[i, i, :, :] += onerdm_b_padded_without_diag

    for i in range(n_occ_b):
        twordm_bb_padded[i, i, :, :] += onerdm_b_padded_without_diag
        twordm_bb_padded[:, :, i, i] += onerdm_b_padded_without_diag
        twordm_bb_padded[:, i, i, :] -= onerdm_b_padded_without_diag
        twordm_bb_padded[i, :, :, i] -= onerdm_b_padded_without_diag_T
        twordm_ab_padded[:, :, i, i] += onerdm_a_padded_without_diag

    for i, j in it.product(range(n_occ_a), repeat=2):
        twordm_aa_padded[i, i, j, j] += 1
        twordm_aa_padded[i, j, j, i] -= 1

    for i, j in it.product(range(n_occ_b), repeat=2):
        twordm_bb_padded[i, i, j, j] += 1
        twordm_bb_padded[i, j, j, i] -= 1

    for i, j in it.product(range(n_occ_a), range(n_occ_b), repeat=1):
        twordm_ab_padded[i, i, j, j] += 1

    twordm_aa_padded = twordm_aa_padded.transpose(1, 0, 3, 2)
    twordm_bb_padded = twordm_bb_padded.transpose(1, 0, 3, 2)
    twordm_ab_padded = twordm_ab_padded.transpose(1, 0, 3, 2)

    return (onerdm_a_padded, onerdm_b_padded), (twordm_aa_padded, twordm_ab_padded, twordm_bb_padded)
