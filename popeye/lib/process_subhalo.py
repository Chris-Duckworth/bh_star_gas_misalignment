'''
Process subhalo - coordinate transformations for raw code TNG data.

Chris Duckworth cduckastro@gmail.com
'''

import numpy as np
from astropy.cosmology import Planck15
from time_conversions import snap_to_z
import snapshot as ss

def load_particles_transform_relative(subfind_id, snapnum, parttype, com=False, basePath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output', blen=75000):
    '''Given a suhhalo ID and snapshot, this function returns all of the particles of 
       a certain type. The coordinates returned are box wrapped in code units and then 
       transformed into physical units. Finally the positions and velocities are then      
       defined relative to the centre position of the overall subhalo and in the rest 
       frame of the CoM motion.
       
       Parameters
       ----------
       subfind_id : float/int
           Subfind_ID associated with subhalo at the supplied snapshot
       snapnum : float/int
           Snapshot number for object of interest
       parttype : str
           DM/star/gas particle type
       com : bool
           True/False as to whether to define centre using particle's centre of mass.
           
       Returns
       -------
       rel_pos : numpy.ndarray
           (n1, Ndim) positions defined relative to object centre (either CoM or potential)
       rel_vel : numpy.ndarray
           (n1, Ndim) velocities defined relative to CoM motion of object.

       IMPORTANT: this may produce undesired results if you want to keep the centre 
       position consistent between particle types.
       '''
    
    # Finding redshift for a given snapshot.
    z = snap_to_z(snapnum)
    
    # Loading particles of given type.
    if parttype == 'DM':
        props = ss.loadSubhalo(basePath=basePath, snapNum=snapnum, id=subfind_id, partType=parttype, fields = ['Coordinates', 'Velocities', 'Potential'])
        # wrapping particle coordinates.
        pos_code = box_wrap(props['Coordinates'], blen)
        # transforming to physical coordinates.
        pos_physical, vel_physical = code_to_physical(pos_code, props['Velocities'], z)
        
        # transforming pos and velocities relative to the centre 
        if com == True:
            return transform_relative_to_centre(pos_physical, vel_physical, masses=None, potential=None)
        elif com == False:
            return transform_relative_to_centre(pos_physical, vel_physical, masses=None, potential=props['Potential'])
        else:
            raise AssertionError ('com param must be boolean float')
    
    elif (parttype == 'star') | (parttype == 'gas'):
        props = ss.loadSubhalo(basePath=basePath, snapNum=snapnum, id=subfind_id, partType=parttype, fields = ['Coordinates', 'Velocities', 'Potential', 'Masses'])
        # wrapping particle coordinates.
        pos_code = box_wrap(props['Coordinates'], blen)
        # transforming to physical coordinates.
        pos_physical, vel_physical = code_to_physical(pos_code, props['Velocities'], z)
        
        # transforming pos and velocities relative to the centre 
        if com == True:
            return transform_relative_to_centre(pos_physical, vel_physical, masses=props['Masses'], potential=None)
        elif com == False:
            return transform_relative_to_centre(pos_physical, vel_physical, masses=props['Masses'], potential=props['Potential'])
        else:
            raise AssertionError ('com param must be boolean float')
         
    else:
        raise AssertionError ('Particle type not recognised. DM/star/gas')


def box_wrap(pos_comoving, box_side_length):
    '''This function takes raw comoving (i.e. code units) coordinates (for a given object)
       and wraps them. 
       The first position in the array supplied will be taken as the box side at which all
       positions will be moved to. If the wrapping is not needed then should return the
       same positions.'''
    # Defining reference point - taken as first position in the list.
    centre = pos_comoving[0]

    # Finding relative positions for all other particles to this point. -----------------
    # Distance from origin (as defined by first particle in pos.)
    delt_1 = np.abs(pos_comoving - centre)
    # Finding distance from origin to provided coordinates in other direction (i.e. distance as if wrapped)
    delt_2 = box_side_length - np.abs(pos_comoving - centre) 
    # Finding closest direction of the two.
    delta = np.minimum(delt_1, delt_2)    
    
    # If the distance does not run over a boundary, the vector is returned as normal.
    delta[delt_1 == delta] = (pos_comoving - centre)[delt_1 == delta]
    
    # Based on the closest direction, this transforms the coordinates with appropriate signs.
    bsign = np.ones(delta.shape)
    bsign[centre < pos_comoving] = -bsign[centre < pos_comoving]
    # Applying negative signs for those in negative direction. 
    delta[delt_2 == delta] = bsign[delt_2 == delta] * delta[delt_2 == delta]

    # Returning the positions.
    return centre + delta


def code_to_physical(pos_comoving, vel_comoving, z):
    ''' This function accepts the code units (ckpc/h for pos) and transforms them to
        physical units including Hubble flow for vel.'''

    pos_physical = pos_comoving * 1 / (1 + z) * 1 / Planck15.h
    vel_peculiar = vel_comoving * 1 / np.sqrt(1 + z)
    vel_physical_total = vel_peculiar + Planck15.H(z).value * pos_physical / 1000
    return pos_physical, vel_physical_total


def physical_to_code(pos_physical, vel_physical_total, z):
    ''' This function accepts physical units for pos and vel and transforms 
        them to comoving coordinates. 
        IMPORTANT: This assumes input is the total velocity (i.e. hubble flow included), 
        this will return in code units (i.e. comoving). 
        Make sure that physical pos input are in kpc.'''

    # Using inbuilt astropy function with Planck15 cosmology to calc Hubble parameter.
    vel_physical_peculiar = vel_physical_total - Planck15.H(z).value * pos_physical / 1000 
    vel_comoving = vel_physical_peculiar * np.sqrt(1 + z)
    pos_comoving = pos_physical * (1 + z) * Planck15.h
    return pos_comoving, vel_comoving


def transform_relative_to_centre(pos, vel, masses=None, potential=None):
    '''Given a set of particle pos and vel, this function transforms these to be relative
       to the overall distribution. 
       By default this finds the centre of mass pos and vel for the set and then 
       re-defines the coordinate system relative to this.
       i.e. pos relative to CoM and vel have overall CoM motion removed.
       
       Alternatively you can define the centre position to be relative to the particle 
       with minimum potential value. Simply provide the potential values for each particle.
       Velocities are always returned relative to the CoM motion.
              
       Masses parameter is optional and if not provided the function will assume that the
       particles should be weighted equally in CoM calculation.
       
       IMPORTANT: Make sure box-wrapping is applied before doing this for border objects.
       
       Parameters
       ----------
       pos :  ndarray
           (n1, Ndim) cartesian (physical and box wrapped) coordinates. (kpc)
       vel :  ndarray
           (n1, Ndim) cartesian (physical and box wrapped) velocities. (km/s * kpc)
       masses : ndarray (optional)
           (n1) masses for each particle.
       potential : ndarray (optional)
           (n1) potential measures for each particle. if supplied will return positions 
           relative to particle with minimum potential.
           
       Returns
       -------
       rel_pos : numpy.ndarray
           (n1, Ndim) positions defined relative to object centre (either CoM or potential)
       rel_vel : numpy.ndarray
           (n1, Ndim) velocities defined relative to CoM motion of object.
       '''
    
    # Set equal weighting if masses not supplied.
    if masses is None:
        masses = np.ones(pos.shape[0])
        
    # Selecting centre definition.
    if potential is None:
        pos_cen = np.sum(pos * masses[:, np.newaxis], axis=0) / np.sum(masses)
    else:
        pos_cen = pos[np.argmin(potential)]

    # Finding CoM motion for all particles.
    CoM_vel = np.sum(vel * masses[:, np.newaxis], axis=0) / np.sum(masses)     

    return pos - pos_cen, vel - CoM_vel
