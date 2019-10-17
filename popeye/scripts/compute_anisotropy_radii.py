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
import h5py 

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
	num_effective_radii = np.array([0.5, 1, 2, 3, 4, 5])
	radii = num_effective_radii * half_radius    
	
	# Finding radial distances for all stellar particles and then binning.
	stellar_rad = np.linalg.norm(stellar_pos, axis=1)
	stellar_inds = np.digitize(stellar_rad, radii)
	
	# Finding radial distances for all DM particles and then binning.
	DM_rad = np.linalg.norm(DM_pos, axis=1)
	DM_inds = np.digitize(DM_rad, radii)
	
	# For each of the radial bins, computing velocity anisotropy.
	# output arrays.
	stellar_beta_vel = np.array([])
	stellar_beta_vel_err = np.array([])
	stellar_beta_sigma = np.array([])
	DM_beta_vel = np.array([])
	DM_beta_vel_err = np.array([])
	DM_beta_sigma = np.array([])
	
	for i in np.arange(radii.size):
		try:
			out = velocity_anisotropy.compute_anisotropy(stellar_pos[stellar_inds == i], stellar_vel[stellar_inds == i])
			stellar_beta_vel = np.append(stellar_beta_vel, out[0])
			stellar_beta_vel_err = np.append(stellar_beta_vel_err, out[1])
			stellar_beta_sigma = np.append(stellar_beta_sigma, out[2])
			
		except:
			stellar_beta_vel = np.append(stellar_beta_vel, np.nan)
			stellar_beta_vel_err = np.append(stellar_beta_vel_err, np.nan)
			stellar_beta_sigma = np.append(stellar_beta_sigma, np.nan)
			
		try:
			out = velocity_anisotropy.compute_anisotropy(DM_pos[DM_inds == i], DM_vel[DM_inds == i])
			DM_beta_vel = np.append(DM_beta_vel, out[0])
			DM_beta_vel_err = np.append(DM_beta_vel_err, out[1])
			DM_beta_sigma = np.append(DM_beta_sigma, out[2])
			
		except:
			DM_beta_vel = np.append(DM_beta_vel, np.nan)
			DM_beta_vel_err = np.append(DM_beta_vel_err, np.nan)
			DM_beta_sigma = np.append(DM_beta_sigma, np.nan)
	
	return (subfind, snapnum, num_effective_radii, radii, 
		   stellar_beta_vel, stellar_beta_vel_err, stellar_beta_sigma, 
		   DM_beta_vel, DM_beta_vel_err, DM_beta_sigma)
	
# ---------------------------------------------------------------------------------------
# Computing for all MaNGA-like galaxies.

snapnum = 99
output = []
subfind_ids = tab.subfind_id.values

for i, sub in enumerate(subfind_ids):
	print( str(np.round(i/subfind_ids.shape[0] * 100, 2))+'%')
	try:
		output.append(compute_anisotropy_radii(sub, snapnum))
	except Exception:
		continue

# Unzipping our output object and finding number of columns. 
unpacked_output = list(zip(*output))

# ---------------------------------------------------------------------------------------
# Create hdf5 file to store output to.

hf = h5py.File(filepath+'tng100_mpl8_velocity_anisotropy.hdf5', 'w')

# dimensions equal to number of objects.
hf.create_dataset('subfind_id', data = np.array(unpacked_output[0]) )
hf.create_dataset('snapnum', data = np.array(unpacked_output[1]) )

# dimensions equal to number of objects x number of radii considered (bin edges).
hf.create_dataset('num_effective_radii', data = np.array(unpacked_output[2]) )
hf.create_dataset('physical_radii_kpc', data = np.array(unpacked_output[3]) )

# dimensions equal to number of objects x number of radii considered (bin edges).
# velocity anisotropy defined as between bin edges defined in num_effective_radii. 
# The first value corresponds to below the first value/radius.
hf.create_dataset('stellar_beta_vel', data = np.array(unpacked_output[4]) )
hf.create_dataset('stellar_beta_vel_err', data = np.array(unpacked_output[5]) )
hf.create_dataset('stellar_beta_sigma', data = np.array(unpacked_output[6]) )

# dimensions equal to number of objects x number of radii considered (bin edges).
# velocity anisotropy defined as between bin edges defined in num_effective_radii. 
# The first value corresponds to below the first value/radius.
hf.create_dataset('DM_beta_vel', data = np.array(unpacked_output[7]) )
hf.create_dataset('DM_beta_vel_err', data = np.array(unpacked_output[8]) )
hf.create_dataset('DM_beta_sigma', data = np.array(unpacked_output[9]) )

hf.close()

# ---------------------------------------------------------------------------------------