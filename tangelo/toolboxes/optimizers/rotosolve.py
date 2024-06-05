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

import numpy as np


def extrapolate_expval(theta, m_0, m_minus, m_plus, phi=0.0):
    """Extrapolates the expectation value of an observable
    M with respect to a single parameterized rotation (i.e.
    RX, RY, RZ) with angle theta. The extrapolation uses
    samples taken at the angles phi, phi+pi/2, and phi-pi/2.
    This function uses the formula in Appendix A from
    arXiv:1905.09692, by Mateusz Ostaszewski et al.

    Args:
        theta (float): Gate rotation angle to extrapolate to
        m_0 (float): Expectation value of M mat angle phi
        m_minus (float): Expectation value of M mat angle phi - pi/2
        m_plus (float): Expectation value of M mat angle phi + pi/2
        phi (float, optional): Angle of phi. Defaults to 0.0

    Returns:
        float: The expectation value of M estimated for theta.
    """
    a = 0.5*np.sqrt((2 * m_0 - m_plus - m_minus)**2 + (m_plus - m_minus)**2)
    b = np.arctan2(2 * m_0 - m_plus - m_minus, m_plus - m_minus) - phi
    c = 0.5*(m_plus + m_minus)

    return a*np.sin(theta + b) + c


def rotosolve_step(func, var_params, i, *func_args, phi=0.0, m_phi=None):
    """Gradient free optimization step using specific points to
    characterize objective function w.r.t to parameter values.
    Based on formulas in arXiv:1905.09692, Mateusz Ostaszewski

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes variational params as input
                and returns a float.
            var_params (list of float): The variational parameters.
            i (int): Index of the variational parameter to update.
            *func_args (tuple): Optional arguments to pass to func.
            phi (float): Optional angle phi for extrapolation (default is 0.0).
            m_phi (float): Optional estimated value of m_phi
        Returns:
            list of floats: Optimal parameters
            float: Estimated optimal value of func
    """

    # Charaterize sinusoid of objective function using specific parameters
    var_params[i] = phi
    m_0 = func(var_params, *func_args) if m_phi is None else m_phi

    var_params[i] = phi + 0.5 * np.pi
    m_plus = func(var_params, *func_args)

    var_params[i] = phi - 0.5 * np.pi
    m_minus = func(var_params, *func_args)

    # Calculate theta_min based on measured values
    theta_min = phi - 0.5 * np.pi - \
        np.arctan2(2. * m_0 - m_plus - m_minus, m_plus - m_minus)

    if theta_min < -np.pi:
        theta_min += 2 * np.pi
    elif theta_min > np.pi:
        theta_min -= 2 * np.pi

    # calculate extrapolated minimum energy estimate:
    m_min_estimate = \
        extrapolate_expval(theta_min, m_0, m_minus, m_plus, phi=phi)

    # Update parameter to theta_min
    var_params[i] = theta_min

    return var_params, m_min_estimate


def rotosolve(func, var_params, *func_args, ftol=1e-5, maxiter=100,
              extrapolate=False):
    """Optimization procedure for parameterized quantum circuits whose
     objective function varies sinusoidally with the parameters. Based
     on the work by arXiv:1905.09692, Mateusz Ostaszewski.

    Args:
        func (function handle): The function that performs energy
            estimation. This function takes variational parameters as input
            and returns a float.
        var_params (list): The variational parameters.
        ftol (float): Convergence threshold.
        maxiter (int): The maximum number of iterations.
        *func_args (tuple): Optional arguments to pass to func.
        extrapolate (bool): If True, the expectation value of func
            extrapolated from previous calls to `rotosolve_step()` will
            be used instead of a function evaluation. This requires
            only two function evaluations per parameter per iteration,
            but may be less stable on noisy devices. If False, three
            evaluations are used per parameter per iteration.

    Returns:
        float: The optimal energy found by the optimizer.
        list of floats: Optimal parameters.
     """
    # Get intial value, and run rotosolve for up to maxiter iterations
    energy_old = func(var_params, *func_args)

    for it in range(maxiter):

        # Update parameters one at a time using rotosolve_step
        energy_est = energy_old
        for i, theta in enumerate(var_params):
            # Optionally re-use the extrapolated energy as m_phi
            if extrapolate:
                var_params, energy_est = \
                    rotosolve_step(func, var_params, i, *func_args,
                                   phi=theta, m_phi=energy_est)
            else:
                var_params, energy_est = \
                    rotosolve_step(func, var_params, i, *func_args)

        energy_new = func(var_params, *func_args)

        # Check if convergence tolerance is met
        if abs(energy_new - energy_old) <= ftol:
            break

        # Update energy value
        energy_old = energy_new

    return energy_new, var_params


def rotoselect_step(func, var_params, var_rot_axes, i, *func_args):
    """Gradient free optimization step using specific points to
    characterize objective function w.r.t to parameterized
    rotation axes and rotation angles. Based on formulas in
    arXiv:1905.09692, Mateusz Ostaszewski

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes variational parameters and
                parameter rotation axes (list of "RX", "RY", "RZ" strings)
                as input and returns a float.
            var_params (list of float): The variational parameters.
            var_rot_axes (list): List of strings ("RX", "RY", or "RZ")
                corresonding to the axis of rotation for each angle in
                the list of variational parameters.
            i (int): Index of the variational parameter to update.
            *func_args (tuple): Optional arguments to pass to func.
        Returns:
            list of floats: Optimal parameters
            list of strs: Optimal rotation axes
    """
    axes = ['RX', 'RY', 'RZ']
    m_axes = np.zeros(3)
    theta_min_axes = np.zeros(3)

    # Evaluate func at phi = 0 (same result for all axes)
    var_params[i] = 0
    m_0 = func(var_params, var_rot_axes, *func_args)

    # Do a rotosolve step for each axis:
    rotosolve_func_args = (var_rot_axes,) + func_args
    for k, axis in enumerate(axes):
        var_rot_axes[i] = axis
        var_params, m_axes[k] = \
            rotosolve_step(func, var_params, i,
                           *rotosolve_func_args)
        theta_min_axes[k] = var_params[i]

    # Select optimal axis yielding minimal value
    k_opt = np.argmin(m_axes)
    var_rot_axes[i] = axes[k_opt]
    var_params[i] = theta_min_axes[k_opt]

    return var_params, var_rot_axes


def rotoselect(func, var_params, var_rot_axes, *func_args, ftol=1e-5,
               maxiter=100):
    """Optimization procedure for parameterized quantum circuits whose
       objective function varies sinusoidally with the parameters. This
       routine differs from `rotosolve` by sampling expectation values
       using the Pauli {X,Y,Z} generators instead of shifted angles of
       rotation. Based on the work by arXiv:1905.09692, Mateusz
       Ostaszewski.

    Args:
        func (function handle): The function that performs energy
            estimation. This function takes variational parameters and
            parameter rotation axes (list of "RX", "RY", "RZ" strings)
            as input and returns a float.
        var_params (list): The variational parameters.
        var_rot_axes (list): List of strings ("RX", "RY", or "RZ")
            corresonding to the axis of rotation for each angle in
            the list of variational parameters.
        ftol (float): Convergence threshold.
        maxiter (int): The maximum number of iterations.
        *func_args (tuple): Optional arguments to pass to func.

    Returns:
        float: The optimal energy found by the optimizer.
        list of floats: Optimal parameters.
        list of strings: Optimal rotation axes.
    """
    # Check parameters and rotation axes are the same length:
    assert len(var_params) == len(var_rot_axes)

    # Get intial value, and run rotosolve for up to maxiter iterations
    energy_old = func(var_params, var_rot_axes, *func_args)
    for it in range(maxiter):

        # Update parameters one at a time using rotosolve_step
        for i in range(len(var_params)):
            var_params, var_rot_axes = \
                rotoselect_step(func, var_params, var_rot_axes, i, *func_args)
        energy_new = func(var_params, var_rot_axes, *func_args)

        # Check if convergence tolerance is met
        if abs(energy_new - energy_old) <= ftol:
            break

        # Update energy value
        energy_old = energy_new

    return energy_new, var_params, var_rot_axes
