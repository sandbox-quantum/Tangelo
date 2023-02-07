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

"""Module containing functions to manipulate molecular coefficient arrays."""

import numpy as np


def spatial_from_spinorb(one_body_coefficients, two_body_coefficients):
    """Function to reverse openfermion.chem.molecular_data.spinorb_from_spatial.

    Args:
        one_body_coefficients: One-body coefficients (array of 2N*2N, where N
            is the number of molecular orbitals).
        two_body_coefficients: Two-body coefficients (array of 2N*2N*2N*2N,
            where N is the number of molecular orbitals).

    Returns:
        (array of floats, array of floats): One- and two-body integrals (arrays
            of N*N and N*N*N*N elements, where N is the number of molecular
            orbitals.
    """
    # Get the number of MOs = number of SOs / 2.
    n_mos = one_body_coefficients.shape[0] // 2

    # Initialize Hamiltonian integrals.
    one_body_integrals = np.zeros((n_mos, n_mos), dtype=complex)
    two_body_integrals = np.zeros((n_mos, n_mos, n_mos, n_mos), dtype=complex)

    # Loop through coefficients.
    for p in range(n_mos):
        for q in range(n_mos):
            # Populate 1-body integrals.
            one_body_integrals[p, q] = one_body_coefficients[2*p, 2*q]
            # Continue looping to prepare 2-body integrals.
            for r in range(n_mos):
                for s in range(n_mos):
                    two_body_integrals[p, q, r, s] = two_body_coefficients[2*p, 2*q+1, 2*r+1, 2*s]

    return one_body_integrals, two_body_integrals
