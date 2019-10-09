'''Quick function to test the estimation of anisotropy calculations.'''

import numpy as np
import groupcat as gc
import process_subhalo
import pandas as pd 
import velocity_anisotropy
import fractional_radii

# selecting a random subhalo. (i.e. one defined in manga sample.)
# ---------------------------------------------------------------------------------------

filepath = '/home/cduckworth/bh_star_gas_misalignment/popeye/catalogues/'
basepath = '/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output/'
treepath = '/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/postprocessing/trees/SubLink/'

# loading in tree
tree = readtreeHDF5.TreeDB(treepath)
# loading in subfind_ids to consider
tab = pd.read_csv(filepath+'tng100_mpl8_pa_info_v0.1.csv')

# ---------------------------------------------------------------------------------------

subfind = tab.subfind_id.values[5000]
snapnum = 99

# ---------------------------------------------------------------------------------------

pos, vel = process_subhalo.load_particles_transform_relative(subfind, snapnum, 'star', com=False, basePath=basepath, blen=75000)

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