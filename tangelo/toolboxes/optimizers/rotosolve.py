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


def rotosolve_step(func, var_params, i, *func_args):
    """Gradient free optimization step using specific points to
    characterize objective function w.r.t to parameter values. Based on
    formulas in arXiv:1905.09692, Mateusz Ostaszewski

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes variational params as input
                and returns a float.
            var_params (list of float): The variational parameters.
            i (int): Index of the variational parameter to update.
            *func_args (tuple): Optional arguments to pass to func.
        Returns:
            list of floats: Optimal parameters.
    """

    # Charaterize sinusoid of objective function using specific parameters
    var_params[i] = 0
    m_1 = func(var_params, *func_args)

    var_params[i] = 0.5 * np.pi
    m_2 = func(var_params, *func_args)

    var_params[i] = -0.5 * np.pi
    m_3 = func(var_params, *func_args)

    # Calculate theta_min based on measured values
    theta_min = -0.5 * np.pi - np.arctan2(2. * m_1 - m_2 - m_3, m_2 - m_3)

    if theta_min < -np.pi:
        theta_min += 2 * np.pi
    elif theta_min > np.pi:
        theta_min -= 2 * np.pi

    # Update parameter to theta_min
    var_params[i] = theta_min

    return var_params


def rotosolve(func, var_params, *func_args, ftol=1e-5, maxiter=100):
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

    Returns:
        float: The optimal energy found by the optimizer.
        list of floats: Optimal parameters.
     """
    # Get intial value, and run rotosolve for up to maxiter iterations
    energy_old = func(var_params, *func_args)
    for it in range(maxiter):
        # Update parameters one at a time using rotosolve_step
        for i in range(len(var_params)):
            var_params = rotosolve_step(func, var_params, i, *func_args)
        energy_new = func(var_params, *func_args)

        # Check if convergence tolerance is met
        if abs(energy_new - energy_old) <= ftol:
            break

        # Update energy value
        energy_old = energy_new

    return energy_new, var_params
