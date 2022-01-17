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
import scipy.optimize as sp


def diis(energies, coeffs):
    """
    DIIS extrapolation, originally developped by Pulay in
    Chemical Physics Letters 73, 393-398 (1980)

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    return extrapolation(energies, coeffs, 1)


def richardson(energies, coeffs, estimate_exp=False):
    """
    General, DIIS-like extrapolation procedure as found in
    Nature 567, 491-495 (2019) [arXiv:1805.04492]

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    if estimate_exp is False:
        return richardson_analytical(energies, coeffs)
    else:
        return richardson_with_exp_estimation(energies, coeffs)


def extrapolation(energies, coeffs, N=None):
    """
    General, DIIS-like extrapolation procedure as found in
    Nature 567, 491-495 (2019) [arXiv:1805.04492]

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors
        N (int): Taylor expanion order; N=None for Richardson extrapolation (order determined from number of datapoints), N=1 for DIIS extrapolation

    Returns:
        float: Extrapolated energy
    """
    n = len(coeffs)
    if N is None:
        N = n-1
    Eh = np.array(energies)
    ck = np.array(coeffs)
    ck = np.array([ck**k for k in range(1, N+1)])
    B = np.ones((n+1, n+1))
    B[n, n] = 0
    B[:n, :n] = ck.T @ ck
    b = np.zeros(n+1)
    b[n] = 1
    x = np.linalg.lstsq(B, b, rcond=None)[0]
    x = x[:-1]
    return np.dot(Eh, x)


def richardson_analytical(energies, coeffs):
    """
    Richardson extrapolation exlicit result as found in
    Phys. Rev. Lett. 119, 180509 [arXiv:1612.02058]

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    Eh = np.array(energies)
    ck = np.array(coeffs)
    x = np.array([np.prod(ai/(a - ai)) for i, a in enumerate(ck)
                      for ai in [np.delete(ck, i)]])
    return np.dot(Eh, x)


def richardson_with_exp_estimation(energies, coeffs):
    """
    Richardson extrapolation by recurrence, with exponent estimation

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors

    Returns:
        float: Extrapolated energy
    """
    n = len(coeffs)
    p = 1
    Eh = np.array(energies)
    c = np.array(coeffs)
    ck = np.array(coeffs)
    p_old = 0
    dp = 0
    for i in range(n-1):
        ti = ck[0]/ck[1]
        si = ck[0]/ck[2]
        if ((n > 2) & (i < (n-2))):
            def f(k):
                tk = np.sign(ti)*np.abs(ti)**k
                sk = np.sign(si)*np.abs(si)**k
                A1 = (tk*Eh[1] - Eh[0])/(tk - 1)
                A2 = (sk*Eh[2] - Eh[0])/(sk - 1)
                return (A1 - A2)**2
            p = sp.minimize(f, p+1, method='BFGS', options={'disp': False}).x[0]
            if (i == 0):
                ck = c**p
        else:
            break
        for j in range(n-i-1):
            ti = (ck[j]/ck[j+1])
            if (i > 0):
                dp = p - p_old
                if (dp == 0):
                    break
                ck[j] = ck[j]*(c[j]**dp - c[j+1]**dp)/(ti - 1)
                ti = (ck[j]/ck[j+1])
            else:
                ck[j] = ck[j]*(c[j] - c[j+1])/(ti - 1)
            Eh[j] = (ti*Eh[j+1] - Eh[j])/(ti - 1)
        p_old = p
    return(Eh[0])
