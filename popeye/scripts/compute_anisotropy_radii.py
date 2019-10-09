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

def compute_radial_anisotropy(subfind, snapnum):
    # First of all loading in stellar positions and velocities (relative to whole object).
    stellar_pos, stellar_vel = process_subhalo.load_particles_transform_relative(subfind, snapnum, 'star', com=False, basePath=basepath, blen=75000)
    # loading in masses for all of the particles.
    masses = ss.loadSubhalo(basePath, 99, id=subfind, partType='star', fields = ['Masses']
    








# ---------------------------------------------------------------------------------------
# calculating mass in radii.

percentiles = np.array([50])
radii = fractional_radii.mass_enclosed_radii(pos, percentiles, weights=None)

radii = np.array([0.5, 1, 2, 3, 4, 5, np.inf]) * radii

# computing radius (spherical)
rad = np.linalg.norm(pos, axis=1)

# finding which of particles fall within specific bin.
inds = np.digitize(rad, radii)

for i in np.arange(radii.size):
    radii[i], velocity_anisotropy.compute_anisotropy(pos[inds == i], vel[inds == i])

pos, vel = process_subhalo.load_particles_transform_relative(subfind, snapnum, 'DM', com=False, basePath=basepath, blen=75000)

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