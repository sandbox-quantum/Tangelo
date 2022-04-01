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

def rotosolve_step(func, var_params,i):
        #Gradient free optimization step - choose specific points to characterize
        #objective function w.r.t to parameter values
        var_params[i] = 0
        m_1 = func(var_params)
        
        var_params[i] = 0.5 * np.pi
        m_2 = func(var_params)
        
        var_params[i] = -0.5 * np.pi
        m_3 = func(var_params)
        
        #calculate theta_min based on measured values
        theta = -0.5 * np.pi - np.arctan2(2. * m_1 - m_2 - m_3, m_2 - m_3)
        
        if theta < -np.pi:
            theta += 2 * np.pi
        elif theta > np.pi:
            theta -= 2 * np.pi     
            
        var_params[i] = theta  
        return var_params

def rotosolve(func, var_params, tolerance=1e-5, max_iterations=50):
         """Function to optimize parameterized quantum circuits whose objective
         function varies sinusoidally with the parameters. Based on the work by
         arXiv:1905.09692, Mateusz Ostaszewski.

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes var_params as input and returns
                a float.
            var_params (list): The variational parameters (float64).
            tolerance (float): Convergence threshold (float64).
            max_iterations (int): The variational parameters (int).

        Returns:
            float: The optimal energy found by the optimizer.
            list of floats: Optimal parameters.
   
         """
         energy_old=func(var_params)           
         for it in range(0, max_iterations):            
            for i, theta in enumerate(var_params):
                  var_params=rotosolve_step(func, var_params, i)
            energy_new=func(var_params)     
            if abs(energy_new-energy_old) <= tolerance:
                break
            energy_old=energy_new
         return energy_new, var_params

        
    
