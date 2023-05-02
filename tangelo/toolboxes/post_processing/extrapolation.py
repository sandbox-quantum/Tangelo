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

import numpy as np
import scipy.optimize as sp


def diis(coeffs, energies, stderr=None):
    """
    DIIS extrapolation, originally developed by Pulay in
    Chemical Physics Letters 73, 393-398 (1980)

    Args:
        coeffs (array-like): Noise rate amplification factors
        energies (array-like): Energy expectation values for amplified noise rates
        stderr (array-like, optional): Energy standard error estimates

    Returns:
        float: Extrapolated energy
        float: Error estimation for extrapolated energy
    """
    return extrapolation(coeffs, energies, stderr, 1)


def richardson(coeffs, energies, stderr=None, estimate_exp=False):
    """
    General, DIIS-like extrapolation procedure as found in
    Nature 567, 491-495 (2019) [arXiv:1805.04492]

    Args:
        coeffs (array-like): Noise rate amplification factors
        energies (array-like): Energy expectation values for amplified noise rates
        stderr (array-like, optional): Energy standard error estimates
        estimate_exp (bool, optional): Choose to estimate exponent in the Richardson method. Default is False.

    Returns:
        float: Extrapolated energy
        float: Error estimation for extrapolated energy
    """
    if not estimate_exp:
        # If no exponent estimation, run the direct Richardson solution
        return richardson_analytical(coeffs, energies, stderr)
    else:
        # For exponent estimation run the Richardson recursive algorithm
        return richardson_with_exp_estimation(coeffs, energies, stderr)


def extrapolation(coeffs, energies, stderr=None, taylor_order=None):
    """
    General, DIIS-like extrapolation procedure as found in
    Nature 567, 491-495 (2019) [arXiv:1805.04492]

    Args:
        coeffs (array-like): Noise rate amplification factors
        energies (array-like): Energy expectation values for amplified noise rates
        stderr (array-like, optional): Energy standard error estimates
        taylor_order (int, optional): Taylor expansion order;
            None for Richardson extrapolation (order determined from number of datapoints),
            1 for DIIS extrapolation

    Returns:
        float: Extrapolated energy
        float: Error estimation for extrapolated energy
    """
    n = len(coeffs)
    if taylor_order is None:
        # Determine the expansion order in case of Richardson extrapolation
        taylor_order = n-1
    Eh = np.array(energies)
    coeffs = np.array(coeffs)

    # Set up the linear system matrix
    ck = np.array([coeffs**k for k in range(1, taylor_order+1)])
    B = np.ones((n+1, n+1))
    B[n, n] = 0
    B[:n, :n] = ck.T @ ck

    # Set up the free coefficients
    b = np.zeros(n+1)
    b[n] = 1  # For the Lagrange multiplier

    # Solve  the DIIS equations by least squares
    x = np.linalg.lstsq(B, b, rcond=None)[0][:-1]

    if stderr is None:
        return np.dot(Eh, x)
    else:
        stderr = np.array(stderr)
        return np.dot(Eh, x), np.sqrt(np.dot(stderr**2, x**2))


def richardson_analytical(coeffs, energies, stderr=None):
    """
    Richardson extrapolation explicit result as found in
    Phys. Rev. Lett. 119, 180509 [arXiv:1612.02058] (up to sign difference)

    Args:
        coeffs (array-like): Noise rate amplification factors
        energies (array-like): Energy expectation values for amplified noise rates
        stderr (array-like, optional): Energy standard error estimates

    Returns:
        float: Extrapolated energy
        float: Error estimation for extrapolated energy
    """
    Eh = np.array(energies)
    ck = np.array(coeffs)
    x = np.array([np.prod(ai / (ai - a)) for i, a in enumerate(ck) for ai in [np.delete(ck, i)]])

    if stderr is None:
        return np.dot(Eh, x)
    else:
        stderr = np.array(stderr)
        return np.dot(Eh, x), np.sqrt(np.dot(stderr**2, x**2))


def richardson_with_exp_estimation(coeffs, energies, stderr=None):
    """
    Richardson extrapolation by recurrence, with exponent estimation

    Args:
        energies (array-like): Energy expectation values for amplified noise rates
        coeffs (array-like): Noise rate amplification factors
        stderr (array-like, optional): Energy standard error estimates

    Returns:
        float: Extrapolated energy
        float: Error estimation for extrapolated energy
    """
    n = len(coeffs)
    Eh = np.array(energies)
    c = np.array(coeffs)
    ck = np.array(coeffs)
    if stderr is not None:
        stderr = np.array(stderr)
    p, p_old = 1, 0

    # Define a helper function for exponent optimization
    def energy_diff(k, ti, si):
        tk = np.sign(ti)*np.abs(ti)**k
        sk = np.sign(si)*np.abs(si)**k
        Et = (tk*Eh[1] - Eh[0])/(tk - 1)
        Es = (sk*Eh[2] - Eh[0])/(sk - 1)
        return (Et - Es)**2

    # Run the Richardson algorithm with exponent optimization
    for i in range(n-1):
        ti = ck[0]/ck[1]
        si = ck[0]/ck[2]
        if ((n > 2) and (i < (n-2))):
            # Minimize the energy difference to determine the optimal exponent
            p = sp.minimize(energy_diff, p+1, args=(ti, si), method='BFGS', options={'disp': False}).x[0]
            if (i == 0):
                ck = c**p
        else:
            break

        # Run the main Richardson loop
        for j in range(n-i-1):
            ti = (ck[j]/ck[j+1])
            if (i > 0):
                dp = p - p_old
                if (np.isclose(dp, 0)):
                    break
                ck[j] = ck[j]*(c[j]**dp - c[j+1]**dp)/(ti - 1)
                ti = (ck[j]/ck[j+1])
            else:
                ck[j] = ck[j]*(c[j] - c[j+1])/(ti - 1)

            Eh[j] = (ti*Eh[j+1] - Eh[j])/(ti - 1)
            if stderr is not None:
                stderr[j] = np.sqrt(((ti*stderr[j+1])**2 + stderr[j]**2)/(ti - 1)**2)

        p_old = p

    if stderr is None:
        return Eh[0]
    else:
        return Eh[0], stderr[0]
