'''
Velocity anisotropy (radial/tangential) calculations for particles in cartesian coordinates.

Chris Duckworth cduckastro@gmail.com
'''

import numpy as np 

def compute_anisotropy(radial, velocity):
    '''This function calculates the anisotropy for the complete set of particles defined
       relative to the halo. Returns both beta defined on magnitude of velocities and 
       dispersion.
       
       Inputs:
           - radial: Rest frame cartesian position vector. dim:3xN [X,Y,Z] 
             defined relative to halo/subhalo centre.
           - velocity: Rest frame cartesian velocity vector. dim:3xN [Vx,Vy,Vz]  
           
       Output:
           - beta_vel (velocity ratio) = 1 - v_tan**2 / v_rad**2 
           - beta_vel_err (velocity ratio error)
           - beta_sigma (dispersion ratio) = 1 - sigma_tan*2 / sigma_rad**2
       '''

    # this calculation fails if centered on individual particle (like with potential 
    # minimum centers). fail-safing this by removing particles 0 distance from centre.
    mask = np.linalg.norm(radial, axis=1) != 0
    radial = radial[mask]
    velocity = velocity[mask]

    # Finding projected velocity vector along radial direction for all particles.
    v_rad = np.vstack(np.einsum('ij,ij->i', velocity, radial) / np.einsum('ij,ij->i', radial, radial))*radial  
    
    # Finding the tangential velocity vector using velocity = v_rad + v_tan
    v_tan = velocity - v_rad
    
    # Finding magnitudes of tangential and radial velocity vectors.
    v_rad_abs = np.linalg.norm(v_rad, axis=1)
    v_tan_abs = np.linalg.norm(v_tan, axis=1)

    # Calculating the mean the velocities squared. 
    vsm_rad, vsm_tan = np.mean(v_rad_abs**2), np.mean(v_tan_abs**2)
    
    # Calculating the standard deviation for error propagation.
    vs_std_rad, vs_std_tan = np.std(v_rad_abs**2), np.std(v_tan_abs**2)
    N = v_rad_abs.shape[0]
    
    # Calculating the dispersion in radial and tangential directions.
    disp_rad, disp_tan = np.std(v_rad_abs), np.std(v_tan_abs)
        
    # Calculating anisotropy for all particles through two methods.
    beta_vel_av = 1 - vsm_tan / (2 * vsm_rad)
    beta_disp_av = 1 - disp_tan**2/(2*disp_rad**2)

    # Estimating errorbars on beta_vel measurement.
    beta_vel_err = (vs_std_tan**2 / N ) * 1 / (4*vsm_rad**2) + (vs_std_rad**2 / N ) * vsm_tan**2 / (4*vsm_rad**4) 
                    
    return beta_vel_av, beta_vel_err, beta_disp_av