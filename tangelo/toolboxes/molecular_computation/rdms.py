# Copyright 2021 1QB Information Technologies Inc.
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

from tangelo.linq.helpers import filter_bases
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.toolboxes.post_processing import Histogram, aggregate_histograms
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.molecular_computation.molecule import spatial_from_spinorb
from tangelo.linq.helpers.circuits import pauli_of_to_string


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


def compute_rdms(ferm_ham, exp_data, mapping, up_then_down):
    """
    Computes the 1- and 2-RDM and their spin-summed versions 
    using a Molecule object and frequency data either in the form of a 
    classical shadow or a dictionary of frequency histograms.

    Args:
        ferm_ham (FermionicOperator): Fermionic operator with n_spinorbitals and n_electrons defined
        exp_data (ClassicalShadow or dict): classical shadow or a dictionary of expectation values of qubit terms
        mapping: qubit mapping
        up_then_down: spin ordering for the mapping

    Returns:
        complex array: 1-RDM
        complex array: 2-RDM
        complex array: spin-summed 1-RDM
        complex array: spin-summed 2-RDM
    """
    onerdm = np.zeros((ferm_ham.n_spinorbitals,) * 2, dtype=complex)
    twordm = np.zeros((ferm_ham.n_spinorbitals,) * 4, dtype=complex)
    onerdm_spinsum = np.zeros((ferm_ham.n_spinorbitals//2,) * 2, dtype=complex)
    twordm_spinsum = np.zeros((ferm_ham.n_spinorbitals//2,) * 4, dtype=complex)

    exp_vals = {}

    # Go over all terms in fermionic Hamiltonian
    for term in ferm_ham.terms:
        length = len(term)

        if not term:
            continue
        
        # Fermionic term with a prefactor of 1.0.
        fermionic_term = FermionOperator(term, 1.0)

        qubit_term = fermion_to_qubit_mapping(fermion_operator = fermionic_term,
                                            n_spinorbitals = ferm_ham.n_spinorbitals,
                                            n_electrons = ferm_ham.n_electrons,
                                            mapping = mapping,
                                            up_then_down = up_then_down,
                                            spin = ferm_ham.spin)
        qubit_term.compress()

        # Loop to go through all qubit terms.
        eigenvalue = 0.
        
        if type(exp_data) == ClassicalShadow:
            for qterm, coeff in qubit_term.terms.items():
                if coeff.real != 0:
                    # Change depending on if it is randomized or not.
                    eigenvalue += exp_data.get_term_observable(qterm, coeff)

        if type(exp_data) == dict:
            for qterm, coeff in qubit_term.terms.items():
                if coeff.real != 0:

                    try:
                        exp_vals[qterm]
                    except KeyError:
                        if qterm:
                            ps = pauli_of_to_string(qterm, ferm_ham.n_spinorbitals // 2)  # not sure about the number
                            exp_vals[qterm] = aggregate_histograms(*[Histogram(exp_data[basis]) for basis in filter_bases(ps, exp_data.keys())]).get_expectation_value(qterm, 1.)
                        else:
                            continue

                    exp_val = exp_vals[qterm] if qterm else 1.
                    eigenvalue += coeff * exp_val

        # Put the values in np arrays (differentiate 1- and 2-RDM)
        if length == 2:
            iele, jele = (int(ele[0]) for ele in tuple(term[0:2]))
            onerdm[iele, jele] += eigenvalue
        elif length == 4:
            iele, jele, kele, lele = (int(ele[0]) for ele in tuple(term[0:4]))
            twordm[iele, lele, jele, kele] += eigenvalue

    # Construct spin-summed 1-RDM.
    for i, j in it.product(range(onerdm.shape[0]), repeat=2):
        onerdm_spinsum[i//2, j//2] += onerdm[i, j]

    # Construct spin-summed 2-RDM.
    for i, j, k, l in it.product(range(twordm.shape[0]), repeat=4):
        twordm_spinsum[i//2, j//2, k//2, l//2] += twordm[i, j, k, l]

    return onerdm, twordm, onerdm_spinsum, twordm_spinsum


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
