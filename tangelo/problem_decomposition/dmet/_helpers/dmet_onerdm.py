# Copyright SandboxAQ 2021-2024.
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

"""Construct one-particle RDM for DMET calculation.

Construction of the one-particle reduced density matrix (RDM) is done here.
"""

import numpy as np


def dmet_low_rdm_rhf(active_fock, n_active_electrons):
    """Construct the one-particle RDM from low-level calculation.

    Args:
        active_fock (numpy.array): Fock matrix from low-level calculation
            (float64).
        n_active_electrons (int): Number of electrons in the entire system.

    Returns:
        numpy.array: One-particle RDM of the low-level calculation (float64).
    """

    # Extract the occupied part of the one-particle RDM
    num_occ = n_active_electrons // 2
    e, c = np.linalg.eigh(active_fock)
    new_index = e.argsort()
    c = c[:, new_index]
    onerdm = np.dot(c[:, : num_occ], c[:, : num_occ].T) * 2

    return onerdm


def dmet_low_rdm_rohf_uhf(active_fock_alpha, active_fock_beta, n_active_alpha, n_active_beta):
    """Construct the one-particle RDM from low-level calculation.

    Args:
        active_fock_alpha (numpy.array): Fock matrix from low-level calculation
            (float64).
        active_fock_beta (numpy.array): Fock matrix from low-level calculation
            (float64).
        n_active_alpha (int): Number of alpha electrons.
        n_active_beta (int): Number of beta electrons.

    Returns:
        onerdm (numpy.array): One-particle RDM of the low-level calculation
            (float64).
    """

    e, c = np.linalg.eigh(active_fock_alpha)
    new_index = e.argsort()
    c = c[:, new_index]
    onerdm_alpha = np.dot(c[:, :int(n_active_alpha)], c[:, :int(n_active_alpha)].T)

    e, c = np.linalg.eigh(active_fock_beta)
    new_index = e.argsort()
    c = c[:, new_index]
    onerdm_beta = np.dot(c[:, :int(n_active_beta)], c[:, :int(n_active_beta)].T)

    return onerdm_alpha + onerdm_beta


def dmet_fragment_rdm(t_list, bath_orb, core_occupied, n_active_electrons):
    """Construct the one-particle RDM for the core orbitals.

    Args:
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        bath_orb (numpy.array): The bath orbitals (float64).
        core_occupied (numpy.array): Core occupied part of the MO coefficients
            (float64).
        n_active_electrons (int): Number of electrons in the entire system.

    Returns:
        int: Number of orbitals for fragment calculation.
        int: Number of electrons for fragment calulation.
        numpy.array: Core part of the one-particle RDM (float64).
    """

    # Obtain number of active orbitals
    number_orbitals = t_list[0] + t_list[1]

    # Round the values above or below threshold
    for i, core in enumerate(core_occupied):
        if (core < 0.01):
            core_occupied[i] = 0.0
        elif (core > 1.99):
            core_occupied[i] = 2.0

    # Define the number of electrons in the fragment
    number_ele_temp = np.sum(core_occupied)
    number_electrons = int(round(n_active_electrons - number_ele_temp))

    # Obtain the one particle RDM for the fragment (core)
    core_occupied_onerdm = bath_orb @ np.diag(core_occupied) @ bath_orb.T

    return number_orbitals, number_electrons, core_occupied_onerdm
