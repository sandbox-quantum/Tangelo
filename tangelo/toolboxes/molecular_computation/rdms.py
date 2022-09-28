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

import numpy as np

from tangelo.toolboxes.molecular_computation.molecule import spatial_from_spinorb


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
