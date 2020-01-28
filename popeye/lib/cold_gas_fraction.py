'''
gas_fractions
These functions compute the gas temperature and returns the mass (and fraction) and returns
the fraction within 
'''

import numpy as np
import snapshot as ss

def radial_pos(cen,sat,blen):
	'''
	radial_pos : returns the box-wrapped radial positions relative to a cente.
	'''
	del1 = np.abs(sat - cen)
	del2 = blen-np.abs(sat - cen)
	delt = np.minimum(del1,del2)
	
	# If the distance does not run over a boundary, the vector is returned as normal.
	delt[del1 == delt] = (sat - cen)[del1 == delt]
	
	# To find the vector between the points across the boundary it gets a little trickier.
	# Again we use the magnitude of the distance found earlier.
	# Depending on which side the cen/sat of the boundary, a sign is applied. 
	
	# Array of signs
	bsign = np.ones(delt.shape)
	bsign[cen < sat] = -bsign[cen < sat]
	
	# Applying signs to relevant points
	delt[del2 == delt] = bsign[del2 == delt]*delt[del2 == delt]
	return delt


def compute_fraction_2re(subfind_id, snapnum, radius, centre, basePath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output'):
	'''
	Function that returns integrated black hole properties for a given subhalo at a certain
	snapshot.
	
	   Parameters
	   ----------
	   subfind_id : float/int
		   Subfind_ID associated with subhalo at the supplied snapshot
	   snapnum : float/int
		   Snapshot number for object of interest
	   basePath : str
		   Base directory for output of TNG simulation.
	   Tcrit : float
	       Cutoff to determine cold vs hot gas.
	   
	   Returns (naming convention follows individual BH particles in TNG)
	   -------
	   cold_gas_frac : float
	       Fraction of gas mass in subhalo that is below temperature threshold.
	'''
	
	# loading in all gas cells for this subhalo.
	props = ss.loadSubhalo(basePath=basePath, snapNum=snapnum, id=subfind_id, partType='gas', fields = ['Coordinates', 'ElectronAbundance', 'StarFormationRate', 'InternalEnergy', 'Masses'])
	
	# if no gas cells, then returning -inf for values.
	if props['count'] == 0:
		return -np.inf
	
	# making radial selection.
	pos = radial_pos(centre, props['Coordinates'], 75000)
	radii = np.linalg.norm(pos, axis=1)
	radial_mask = (radii <= radius)
	# total mass within radius.
	gas_mass_total_inRad = np.sum(props['Masses'][radial_mask])
	
	# compute gas temperature for all cells
	Xh = 0.76 # hydrogen mass fraction
	mp_cgs = 1.6726231 * 10 ** -24 # proton mass in cgs.
	gamma = 5.0 / 3.0 # adiabatic index
	kb_cgs = 1.38064852 * 10 ** -16 # boltzmann constant in cgs.
	
	mean_molecular_weight = 4 / (1 + 3 * Xh + 4 * Xh * props['ElectronAbundance']) * mp_cgs
	temp = (gamma - 1) * props['InternalEnergy'] / kb_cgs * 10**10 * mean_molecular_weight
	
	# selecting star forming gas or that which meets lower temperature criteria.
	cold_phase_mask = (props['StarFormationRate'] > 0) | (temp < 10**4.5)
	# total cold phase within radius
	gas_mass_cold_inRad = np.sum(props['Masses'][(radial_mask) & (cold_phase_mask)])
	
	return gas_mass_cold_inRad


def compute_fraction_set(subs, snaps, radii, centres, basePath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output'):
	'''
	For a set of subfind_ids defined at the corresponding snapshots, run compute_fraction.
	'''
	return np.array([compute_fraction_2re(subfind_id, snapnum, radius, centre) for subfind_id, snapnum, radius, centre in zip(subs, snaps, radii, centres)])