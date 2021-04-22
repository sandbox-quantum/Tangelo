#   Copyright 2019 1QBit
#   
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

"""Constructing guess orbitals in DMET calculation

Construction of the guess orbitals for fragment 
SCF calculation in DMET calculation is done here.

"""

import scipy
import numpy as np
from functools import reduce

def dmet_fragment_guess(t_list, bath_orb, chemical_potential, norb_high, number_active_electron, active_fock):
    """Construct the guess orbitals.

    Args:
        t_list (list): Number of fragment & bath orbitals (int).
        bath_orb (numpy.array): The bath orbitals (float64).
        chemical_potential (float64): The chemical potential.
        norb_high (int): The number of orbitals in the fragment calculation.
        number_active_electron (int): The number of electrons in the fragment calculation.
        active_fock (numpy.array): The fock matrix from the low-level calculation (float64).

    Returns:
        frag_guess (numpy.array): The guess orbitals (float64).
    """

    # Construct the fock matrix of the fragment (subtract the chemical potential for consistency)
    fock_fragment = reduce(np.dot, (bath_orb[ : , : norb_high].T, active_fock, bath_orb[ : , : norb_high]))
    norb = t_list[0]
    if(chemical_potential != 0):
        for i in range(norb):
            fock_fragment[i, i] -= chemical_potential

    # Diagonalize the fock matrix and get the eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(fock_fragment)
    eigenvectors = eigenvectors[ : , eigenvalues.argsort()]

    # Extract the eigenvectors of the occupied orbitals as the guess orbitals
    frag_guess = np.dot(eigenvectors[ :, : int(number_active_electron/2)], eigenvectors[ :, : int(number_active_electron/2)].T) * 2
    
    return frag_guess

