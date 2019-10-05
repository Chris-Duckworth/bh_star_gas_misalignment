'''
fractional_radii - given a distribution of particles this returns the spherical radius 
enclosing X % of them.
'''

import numpy as np 

def mass_enclosed_radii(pos, percentiles, weights=None):
    '''
    Given a set of n particle positions (n, 3D) (centered on origin) and a set of 
    percentiles so that 50th percentile returns half mass radius, 99th percentile returns
    the radius enclosing 99% of the mass and so on, this will return the radii that these
    percentiles correspond to (in particle units).
    Option to supply weights (i.e. masses) to particles.
    
    This is calculated by sorting the particles on radius and finding the cumulative mass
    sum. It returns the radius of the particle that is closest to the percentile mass in 
    the cumulative sum. i.e. no interpolation.
    
    Parameters
    ----------
    pos : array_like
        (n, 3D)
        cartesian coordinates defined relative to origin. please center your object on the
        origin or this will produce nonsense.
    percentiles : float or array_like
        (m) 
        single or set of percentiles to define mass enclose radii. (0-100)
    weights : array_like
        (n)
        Weights associated with each particle to compute mass enclosed. 
        Will assume equal weights if you dont supply any.
    
    
    Returns
    -------
    radii : numpy.ndarray
        (m)
        Set of radii (particle units) that define the radius that contains the percentile 
        of mass.
    '''
    
    percentiles = np.array(percentiles)
    
    # checking that the percentiles are in the correct range.
    if True in (percentiles > 100) | (percentiles < 1):
        raise AssertionError('Make sure you are defining a percentile between 1 and 100!')
    
    if weights is None:
        # setting equal weights if not supplied.
        weights = np.ones(pos.shape[0])
    
    # calculating radii for each of the particles defined.
    rad = np.linalg.norm(pos, axis=1)

    # sorting particles radii and returning args.
    rad_argsort = np.argsort(rad)

    # sorting radii and associated weights.
    sorted_rad = rad[rad_argsort]
    sorted_weights = weights[rad_argsort]

    # finding cumulative sum of weights.
    cumsum_weights = np.cumsum(sorted_weights)

    # calculating the enclosed weight percentiles. 
    percentile_weights = np.sum(weights) * percentiles / 100.0

    # list comprehension to find each radius for each percentile.
    return np.array([sorted_rad[np.argmin(np.abs(cumsum_weights - pw))] for pw in percentile_weights])