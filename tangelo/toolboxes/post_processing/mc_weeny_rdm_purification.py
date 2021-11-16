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

import numpy as np
import itertools


def mcweeny_purify_2rdm(rdm2_spin, conv=1.0e-07):
    """Perform 2-RDM purification based on McWeeny"s algorithm. The value
    selected for the convergence criteria should be consistent with the accuracy
    of the 2-RDM provided as input (in particular it should be correlated to the
    number of shots used to compute the RDM values, if a QPU or a shot-based
    quantum circuit simulator was used).

    This algorithm only works on a RDM associated with a 2-electron system.

    Args:
        rdm2_spin (numpy array): The 2-RDM to be purified (in chemistry
            notation).
        conv (float), optional: The convergence criteria for McWeeny"s
            purification.

    Returns:
        numpy.array: One-particle RDM in spatial orbital basis.
        numpy.array: Two-particle RDM in spatial orbital basis (in chemistry
            notation).
    """

    n_spinorbitals = rdm2_spin.shape[0]
    n_spatialorbitals = n_spinorbitals // 2
    rdm1_np = np.zeros((n_spatialorbitals, )*2)
    rdm2_np = np.zeros((n_spatialorbitals, )*4)

    # Transform the 2-RDM in physics notation
    D_matrix2 = np.asarray(rdm2_spin.transpose(0, 2, 1, 3), order="C")

    # Initialize trace of difference to 1.0, repeat McWeeny"s cycle until convergence is met (trace of D-D^{2} ~ 0.)
    diff2 = 1.0
    while abs(diff2) > conv:

        # Update the 2-RDM based on recursion relation
        D2 = np.einsum("pqrs,rsuv->pquv", D_matrix2, D_matrix2)
        D3 = np.einsum("pqrs,rsuv->pquv", D_matrix2, D2)
        D_matrix2 = 3.0 * D2 - 2.0 * D3

        # Compute trace of D-D^{2}
        DD_mat = np.einsum("pqrs,rsuv->pquv", D_matrix2, D_matrix2)
        D_diff = DD_mat - D_matrix2
        diff2 = sum(D_diff[i, j, i, j] for i, j in itertools.product(range(n_spinorbitals), repeat=2))

    # Transform the 2-RDM to chemistry notation
    D_matrix2_final = np.asarray(D_matrix2.transpose(0, 2, 1, 3), order="C")

    # Construct 1-RDM using 2-RDM
    rdm1_np_temp = np.zeros((n_spinorbitals, )*2)
    for i, j, k in itertools.product(range(n_spinorbitals), repeat=3):
        rdm1_np_temp[i, j] += D_matrix2_final[i, j, k, k]
    for i, j in itertools.product(range(n_spinorbitals), repeat=2):
        rdm1_np[i//2, j//2] += rdm1_np_temp[i, j]

    # Construct 2-RDM
    for i, j, k, l in itertools.product(range(n_spinorbitals), repeat=4):
        rdm2_np[i//2, j//2, k//2, l//2] += D_matrix2_final[i, j, k, l]

    return rdm1_np, rdm2_np
