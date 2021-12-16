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
from scipy.optimize import minimize, root


def diis(energies, coeffs):
    """
    DIIS extrapolation

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    n = len(coeffs)
    Eh = np.array(energies)
    ck = np.array(coeffs)
    B = np.ones((n+1, n+1))
    B[n, n] = 0
    B[:n, :n] = ck[:, None] @ ck[None, :]
    b = np.zeros(n+1)
    b[n] = 1
    x = np.linalg.lstsq(B, b, rcond=None)[0]
    x = x[:-1]
    return np.dot(Eh, x)


def richardson(energies, coeffs):
    """
    Richardson extrapolation as found in
    Nature 567, 491-495 (2019) [arXiv:1805.04492]

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    n = len(coeffs)
    Eh = np.array(energies)
    ck = np.array([coeffs**k for k in range(1, n+1)]).sum(axis=0)
    B = np.ones((n+1, n+1))
    B[n, n] = 0
    B[:n, :n] = ck[:, None] @ ck[None, :]
    b = np.zeros(n+1)
    b[n] = 1
    x = np.linalg.lstsq(B, b, rcond=None)[0]
    x = x[:-1]
    return np.dot(Eh, x)
