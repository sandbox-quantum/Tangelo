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

"""Bath orbital construction for DMET calculation.

The construction of the bath orbitals (the orbitals which include the
environment effect from the surrounding part) is done here.
"""

import numpy as np


def dmet_fragment_bath(mol, t_list, temp_list, onerdm_low, virtual_orbital_threshold=1e-13, verbose=False):
    """ Construct the bath orbitals for DMET fragment calculation.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate (The full molecular
            system).
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        temp_list (list): [0] Minimum and [1] maximum number for the active
            orbitals (int).
        onerdm_low (numpy.array): One-particle RDM from the low-level
            calculation (float64).
        virtual_orbital_threshold (float): Occupation threshold for the density
            matrix, used to discard virtual orbitals.
        verbose (bool): Print the orbital occupancy eigenvalues for prototyping
            purposes (setting virtual_orbital_threshold).

    Returns:
        numpy.array: The bath orbitals (float64).
        numpy.array: Orbital energies (float64).
    """

    # Extract the one-particle RDM for the active space
    onerdm_embedded = dmet_onerdm_embed(mol, temp_list, onerdm_low)

    # Diagonalize it
    e, c = np.linalg.eigh(onerdm_embedded)

    # Sort the eigenvectors with the eigenvalues (should be positive unless
    # there is numerical noise, therefore we take the absolute values).
    e = np.abs(e)
    if verbose:
        print(f"\t{e}\n")

    e_sorted, c_sorted = dmet_bath_orb_sort(t_list, e, c, virtual_orbital_threshold)

    # Add the core contribution
    bath_orb, e_core = dmet_add_to_bath_orb(mol, t_list, temp_list, e_sorted, c_sorted)

    return bath_orb, e_core


def dmet_onerdm_embed(mol, temp_list, onerdm_before):
    """ Extract the one particle RDM of the active space.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate (The full molecular
            system).
        temp_list (list): [0] Minimum and [1] maximum number for the active
            orbitals (int).
        onerdm_before (numpy.array): One-particle RDM from the low-level
            calculation (float64).

    Returns:
        numpy.array: Extracted one-particle RDM (float64).
    """

    # Get the number of orbitals
    norbital_total = mol.nao_nr()

    # Reshape the RDM
    onerdm_matrix = np.reshape(onerdm_before, (norbital_total, norbital_total))

    if temp_list[0] == 0:
        # If it is the first fragment, just determine the maximum for extraction
        onerdm_temp = onerdm_matrix[:, temp_list[1]:]
        onerdm_temp3 = onerdm_temp[temp_list[1]:, :]
    else:
        # Determine the minimum and maximum orbitals for extraction
        onerdm_temp = onerdm_matrix[:, : temp_list[0]]
        onerdm_temp2 = onerdm_matrix[:, temp_list[1]:]
        onerdm_temp3 = np.hstack((onerdm_temp, onerdm_temp2))
        onerdm_temp = onerdm_temp3[:temp_list[0], :]
        onerdm_temp2 = onerdm_temp3[temp_list[1]:, :]
        onerdm_temp3 = np.vstack((onerdm_temp, onerdm_temp2))

    return onerdm_temp3


def dmet_bath_orb_sort(t_list, e_before, c_before, virtual_orbital_threshold):
    """ Sort the bath orbitals with the eigenvalues (orbital energies).

    Args:
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        e_before (numpy.array): Orbitals energies before sorting (float64).
        c_before (numpy.array): Coefficients of the orbitals before sorting
            (float64).
        virtual_orbital_threshold (float): Occupation threshold for the density
            matrix, used to discard virtual orbitals.

    Returns:
        numpy.array: Sorted orbital energies (float64).
        numpy.array: Coefficients of the sorted orbitals (float64).
    """

    # Sort the orbital energies (Occupation of 1.0 should come first...)
    new_index = np.maximum(-e_before, e_before - 2.0).argsort()

    # Throw away some orbitals above threshold
    thresh_orb = np.sum(-np.maximum(-e_before, e_before - 2.0)[new_index] > virtual_orbital_threshold)

    # Determine the number of bath orbitals
    norb = min(np.sum(thresh_orb), t_list[0])

    t_list.append(norb)

    # Sort the bath orbitals with its energies
    e_new = e_before[new_index]
    c_new = c_before[:, new_index]

    return e_new, c_new


def dmet_add_to_bath_orb(mol, t_list, temp_list, e_before, c_before):
    """ Add the frozen core part to the bath orbitals.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate (The full molecular
            system).
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        temp_list (list): [0] Minimum and [1] maximum number for the active
            orbitals (int).
        e_before (numpy.array): Orbital energy before addition of frozen core
            (float64).
        c_before (numpy.array): Coefficients of the orbitals before addition of
            frozen core (float64).

    Returns:
        numpy.array: Constructed bath orbitals (float64).
        numpy.array: Orbital energies (float64).
    """

    # Copy the bath orbitals and energies be fore adding the core
    add_e = - e_before[t_list[1]:]
    add_c = c_before[:, t_list[1]:]
    new_index = add_e.argsort()

    # Sort the orbitals based on its energies
    c_before[:, t_list[1]:] = add_c[:, new_index]
    add_e = - add_e[new_index]

    # The orbital energies with core part
    norbital_total = mol.nao_nr()
    e_occupied_core_orbitals = np.hstack((np.zeros([t_list[0] + t_list[1]]), add_e))

    # Add the core part in the orbitals
    for orb in range(0, t_list[0]):
        c_before = np.insert(c_before, orb, 0.0, axis=1)
    i_temp = 0
    for orb_total in range(0, norbital_total):
        if ((orb_total >= temp_list[0]) and (orb_total < temp_list[1])):
            c_before = np.insert(c_before, orb_total, 0.0, axis=0)
            c_before[orb_total, i_temp] = 1.0
            i_temp += 1

    return c_before, e_occupied_core_orbitals
