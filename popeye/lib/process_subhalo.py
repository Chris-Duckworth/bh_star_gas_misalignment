'''
Process subhalo - loading in particle positions and velocities in physical units, relative 
to the CoM/potential minimum for a given subhalo.

Chris Duckworth cduckastro@gmail.com
'''

import numpy as np
from astropy.cosmology import Planck15
from time_conversions import snap_to_z
import snapshot as ss
import coordinate_transforms 

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
        pos_code = coordinate_transforms.box_wrap(props['Coordinates'], blen)
        # transforming to physical coordinates.
        pos_physical, vel_physical = coordinate_transforms.code_to_physical(pos_code, props['Velocities'], z)
        
        # transforming pos and velocities relative to the centre 
        if com == True:
            return coordinate_transforms.transform_relative_to_centre(pos_physical, vel_physical, masses=None, potential=None)
        elif com == False:
            return coordinate_transforms.transform_relative_to_centre(pos_physical, vel_physical, masses=None, potential=props['Potential'])
        else:
            raise AssertionError ('com param must be boolean float')
    
    elif (parttype == 'star') | (parttype == 'gas'):
        props = ss.loadSubhalo(basePath=basePath, snapNum=snapnum, id=subfind_id, partType=parttype, fields = ['Coordinates', 'Velocities', 'Potential', 'Masses'])
        # wrapping particle coordinates.
        pos_code = coordinate_transforms.box_wrap(props['Coordinates'], blen)
        # transforming to physical coordinates.
        pos_physical, vel_physical = coordinate_transforms.code_to_physical(pos_code, props['Velocities'], z)
        
        # transforming pos and velocities relative to the centre 
        if com == True:
            return coordinate_transforms.transform_relative_to_centre(pos_physical, vel_physical, masses=props['Masses'], potential=None)
        elif com == False:
            return coordinate_transforms.transform_relative_to_centre(pos_physical, vel_physical, masses=props['Masses'], potential=props['Potential'])
        else:
            raise AssertionError ('com param must be boolean float')
         
    else:
        raise AssertionError ('Particle type not recognised. DM/star/gas')
