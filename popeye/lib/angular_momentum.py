'''
angular_momentum - returns unit vector direction and magnitude for a given set of relative
defined positions and velocities.
'''

import numpy as np

def compute_angular_momentum(pos, vel, masses=None):
    ''' Given a set of positions and velocities for a set of particles (with associated mass)
        this function returns the specific angular momentum magnitude and direction unit vector.
        
        If masses aren't provided this will assume that the particles should be weighted
        equally.
        
        Parameters
        ----------
        
        pos : ndarray (n1, 3)
            Must already be defined with respect to centre position and scaled by scale 
            factor.
        vel : ndarray (n1, 3)
            Must already be defined with respect to centre motion and scaled by scale 
            factor.
        masses : ndarray (n1) 
            1D masses referring to each particle/cell position. In physical units.
        
        Returns
        -------
        
        magnitude_sJ : float
            Magnitude of specific angular momentum.
        total_ang_mom_unit : ndarray (3)
            unit vector specifying the direction of the angular momentum vector.
            
    '''
    
    # If masses are undefined, all particles are assumed to have a mass of one.
    if masses is None:
        masses = np.ones(pos.shape[0])
    
    # Finding momentum vector for each particle.
    mom = vel * (masses[:,np.newaxis]) 
    
    # Row-by-row cross product. This finds the angular momentum contribution for every particle.
    ang_mom = pos.T[[1,2,0]] * mom.T[[2,0,1]] - pos.T[[2,0,1]] * mom.T[[1,2,0]]
    ang_mom = ang_mom.T
    
    # Finding total angular momentum along X,Y,Z.
    total_ang_mom = np.sum(ang_mom, axis=0)
    
    # Specific angular momentum. dividing by total mass enclosed. Magnitude is then found.
    magnitude_sJ = np.linalg.norm( total_ang_mom / np.sum(masses) ) 
    
    # finding unit vector direction of ang mom
    total_ang_mom_unit = total_ang_mom / np.linalg.norm(total_ang_mom)
    
    return magnitude_sJ, total_ang_mom_unit



def compute_particle_magnitudes(pos, vel, masses=None):
    ''' Given a set of positions and velocities for a set of particles (with associated mass)
        this function returns the specific angular momentum magnitude magnitude of each 
        particle.
        
        If masses aren't provided this will set them to 1.
        
        Parameters
        ----------
        
        pos : ndarray (n1, 3)
            Must already be defined with respect to centre position and scaled by scale 
            factor.
        vel : ndarray (n1, 3)
            Must already be defined with respect to centre motion and scaled by scale 
            factor.
        masses : ndarray (n1) 
            1D masses referring to each particle/cell position. In physical units.
        
        Returns
        -------
        
		ang_mom : ndarray (n1)
			1D array of individual angular momentum for each particle. 
    '''
    
    # If masses are undefined, all particles are assumed to have a mass of one.
    if masses is None:
        masses = np.ones(pos.shape[0])
    
    # Finding momentum vector for each particle.
    mom = vel * (masses[:,np.newaxis]) 
    
    # Row-by-row cross product. This finds the angular momentum contribution for every particle.
    ang_mom = pos.T[[1,2,0]] * mom.T[[2,0,1]] - pos.T[[2,0,1]] * mom.T[[1,2,0]]
    ang_mom = ang_mom.T
    
    # Specific angular momentum. dividing by total mass enclosed. Magnitude is then found.
    magnitude_sJ = np.linalg.norm( ang_mom, axis=1) / masses
    
    print('Not tested for actual science use.')
    return magnitude_sJ
