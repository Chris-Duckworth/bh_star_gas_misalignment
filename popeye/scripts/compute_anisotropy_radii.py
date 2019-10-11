'''
compute_anisotropy_radii - this script computes anisotropy parameters for a subhalo within
a set of annuli (normally in integers of rhalf - stellar).
'''

import numpy as np
import process_subhalo
import pandas as pd 
import velocity_anisotropy
import fractional_radii
import snapshot as ss

# ---------------------------------------------------------------------------------------
# loading in manga-like subhaloes.

filepath = '/home/cduckworth/bh_star_gas_misalignment/popeye/catalogues/'
basepath = '/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output/'


# loading in subfind_ids to consider
tab = pd.read_csv(filepath+'tng100_mpl8_pa_info_v0.1.csv')

# ---------------------------------------------------------------------------------------

def compute_anisotropy_radii(subfind, snapnum):
    # First of all loading in stellar positions and velocities (relative to whole object).
    stellar_pos, stellar_vel = process_subhalo.load_particles_transform_relative(subfind, snapnum, 'star', com=False, basePath=basepath, blen=75000)
    # loading in masses for all of the particles.
    masses = ss.loadSubhalo(basepath, 99, id=subfind, partType='star', fields = ['Masses'])
    # also loading in DM particles.
    DM_pos, DM_vel = process_subhalo.load_particles_transform_relative(subfind, snapnum, 'DM', com=False, basePath=basepath, blen=75000)
        
    # Computing circular r50 stellar. Defining radii as multiples of this.
    percentiles = np.array([50])
    half_radius = fractional_radii.mass_enclosed_radii(stellar_pos, percentiles, weights=masses)
    radii = np.array([0.5, 1, 2, 3, 4, 5, np.inf]) * half_radius    
    
    # Finding radial distances for all stellar particles and then binning.
    stellar_rad = np.linalg.norm(stellar_pos, axis=1)
    stellar_inds = np.digitize(stellar_rad, radii)
    
    # Finding radial distances for all DM particles and then binning.
    DM_rad = np.linalg.norm(DM_pos, axis=1)
    DM_inds = np.digitize(DM_rad, radii)
    
    # For each of the radial bins, computing velocity anisotropy.
    for i in np.arange(radii.size):
        print(radii[i], velocity_anisotropy.compute_anisotropy(stellar_pos[stellar_inds == i], stellar_vel[stellar_inds == i]))
        print(DM_rad[DM_inds == i].shape)
        print(radii[i], velocity_anisotropy.compute_anisotropy(DM_pos[DM_inds == i], DM_pos[DM_inds == i]))
    return




subfind = tab.subfind_id.values[0]
snapnum = 99

velocity_anisotropy.compute_anisotropy(DM_pos[DM_inds == 0], DM_pos[DM_inds == 0])

# ---------------------------------------------------------------------------------------
# calculating mass in radii.

# percentiles = np.array([25, 50, 75, 100])
# radii = fractional_radii.mass_enclosed_radii(pos, percentiles, weights=None)

# computing radius (spherical)
rad = np.linalg.norm(pos, axis=1)

# finding which of particles fall within specific bin.
inds = np.digitize(rad, radii)

for i in np.arange(radii.size):
    radii[i], velocity_anisotropy.compute_anisotropy(pos[inds == i], vel[inds == i])