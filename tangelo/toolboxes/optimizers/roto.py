
def rotosolve_step(func, var_params,i):
        #Simple optimization algorithm for the circuit - closed form minima of a sinusoid
        var_params[i] = 0
        m_1 = func(var_params)
        
        var_params[i] = 0.5 * np.pi
        m_2 = func(var_params)
        
        var_params[i] = -0.5 * np.pi
        m_3 = func(var_params)
        
        theta = -0.5 * np.pi - np.arctan2(2. * m_1 - m_2 - m_3, m_2 - m_3)
        
        if theta < -np.pi:
            theta += 2 * np.pi
        elif theta > np.pi:
            theta -= 2 * np.pi     
            
        var_params[i] = theta  
        return var_params

def rotosolve(func, var_params, tolerance=1e-5,max_iterations=50, verbose=False):
        for it in range(0, max_iterations):
            energy_old=func(var_params) 
            for i, theta in enumerate(var_params):
                  var_params=rotosolve_step(func, var_params, i)
            energy_new=func(var_params)     
            if abs(energy_new-energy_old) <= tolerance:
                break
        return energy_new, var_params

        

    
