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

"""Constructing guess orbitals in DMET calculation

Construction of the guess orbitals for fragment SCF calculation in DMET
calculation is done here.
"""

import scipy
import numpy as np


def dmet_fragment_guess_rhf(t_list, bath_orb, chemical_potential, norb_high, number_active_electron, active_fock):
    """Construct the guess orbitals.

    Args:
        t_list (list): Number of fragment & bath orbitals (int).
        bath_orb (numpy.array): The bath orbitals (float64).
        chemical_potential (float64): The chemical potential.
        norb_high (int): The number of orbitals in the fragment calculation.
        number_active_electron (int): The number of electrons in the fragment
            calculation.
        active_fock (numpy.array): The fock matrix from the low-level
            calculation (float64).

    Returns:
        numpy.array: The guess orbitals (float64).
    """

    # Construct the fock matrix of the fragment (subtract the chemical potential for consistency)
    fock_fragment = bath_orb[:, : norb_high].T @ active_fock @ bath_orb[:, : norb_high]
    norb = t_list[0]
    if(chemical_potential != 0):
        for i in range(norb):
            fock_fragment[i, i] -= chemical_potential

    # Diagonalize the fock matrix and get the eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(fock_fragment)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]

    # Extract the eigenvectors of the occupied orbitals as the guess orbitals
    frag_guess = np.dot(eigenvectors[ :, : int(number_active_electron/2)], eigenvectors[ :, : int(number_active_electron/2)].T) * 2

    return frag_guess


def dmet_fragment_guess_rohf_uhf(t_list, bath_orb, chemical_potential, norb_high,
    number_active_electron, active_fock_alpha, active_fock_beta, n_active_alpha,
     n_active_beta):
    """Construct the guess orbitals.

    Args:
        t_list (list): Number of fragment & bath orbitals (int).
        bath_orb (numpy.array): The bath orbitals (float64).
        chemical_potential (float64): The chemical potential.
        norb_high (int): The number of orbitals in the fragment calculation.
        number_active_electron (int): The number of electrons in the fragment
            calculation.
        active_fock_alpha (numpy.array): The fock matrix from the low-level
            calculation for the alpha electrons (float64).
        active_fock_beta (numpy.array): The fock matrix from the low-level
            calculation for the beta electrons (float64).
        number_active_electron (int): The number of electrons in the fragment
            calculation.
        n_active_alpha (int): The number octive alpha electrons.
        n_active_beta (int): The number octive beta electrons.

    Returns:
        frag_guess (numpy.array): The guess orbitals (float64).
    """

    n_spin = n_active_alpha - n_active_beta
    n_pair = (number_active_electron - n_spin) // 2
    new_alpha = n_pair + n_spin
    new_beta = n_pair

    norb = t_list[0]

    # Construct the fock matrix of the fragment (subtract the chemical potential for consistency)
    fock_fragment = bath_orb[:, : norb_high].T @ active_fock_alpha @ bath_orb[:, : norb_high]
    if(chemical_potential != 0):
        for i in range(norb):
            fock_fragment[i, i] -= chemical_potential

    # Diagonalize the fock matrix and get the eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(fock_fragment)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]

    # Extract the eigenvectors of the occupied orbitals as the guess orbitals
    # Introduce alpha- and beta-electrons
    frag_guess_alpha = np.dot(eigenvectors[:, :int(new_alpha)], eigenvectors[:, :int(new_alpha)].T)

    # Construct the fock matrix of the fragment (subtract the chemical potential for consistency)
    fock_fragment = bath_orb[:, :norb_high].T @ active_fock_beta @ bath_orb[:, :norb_high]
    if(chemical_potential != 0):
        for i in range(norb):
            fock_fragment[i, i] -= chemical_potential

    # Diagonalize the fock matrix and get the eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(fock_fragment)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]

    frag_guess_beta = np.dot(eigenvectors[ :, : int(new_beta)], eigenvectors[ :, : int(new_beta)].T)

    return np.array((frag_guess_alpha, frag_guess_beta)), [new_alpha, new_beta]
