'''
branch properties - returns a variety of properties along the main branch for a given subhalo
'''

import numpy as np
import groupcat as gc
import h5py
from astropy.cosmology import Planck15, z_at_value
import pandas as pd
import bh_luminosity
import bh_params_subhalo
import time_conversions
import cold_gas_fraction

def branch_tabulate(subfind, snapnum, tree, lookback_z, basepath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output'):
    '''
    Function which finds the main branch for a given subhalo (subfind_id, snapnum)
    back to a given redshift (lookback_z).

    Returns a pandas dataframe with:
    - mass history: stellar, gas, halo, black hole
    - black hole luminosity: bolometric and X-ray
    - SFR and gas metallicity
    - black hole propeties: energy growth, mass accreted, local gas density
    - cold gas fraction
    '''

    branch = tree.get_main_branch(snapnum, subfind, keysel=['SubfindID', 'SubhaloMass', 'SubhaloMassType',
                                                            'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloGrNr',
                                                            'SubhaloSFR', 'SubhaloGrNr', 'SubhaloGasMetallicity',
                                                            'SnapNum'])

    # Since branch is constructed from DM only, those halos with zero DM (but stellar comps)
    # may not have ANY branch object.
    if branch == None:
        return pd.DataFrame({})

    # Converting snapnums to redshifts
    branch_z = np.array([time_conversions.snap_to_z(i) for i in branch.SnapNum])
    # Creating mask for maximum lookback snapnum.
    mask = (branch_z <= lookback_z)

    # Finding the halo mass + central/satellite flag.
    halo_mass = np.array([])
    central_flag = np.array([])

    # Looping over all subhalos in main branch. Stopping at z limit.
    for i in np.arange(branch.SnapNum[mask].shape[0]):
        group_info = gc.loadSingle(basepath, branch.SnapNum[i], haloID=branch.SubhaloGrNr[i])
        halo_mass = np.append(halo_mass, group_info['GroupMass'])
        # Finding if current subhalo is a central or satellite in current group.
        if branch.SubfindID[i] == group_info['GroupFirstSub']:
            central_flag = np.append(central_flag, int(1)) # central_flag == 1 if central.
        else:
            central_flag = np.append(central_flag, int(0)) # central_flag == 0 if satellite.

    # Creating root subfind and snapnum columns.
    root_sub = np.full(branch.SubfindID[mask].shape[0], subfind)
    root_snap = np.full(branch.SnapNum[mask].shape[0], snapnum)

    # Converting masses to consistent units.
    halo_mass = halo_mass * 10**10 * (1/Planck15.h)
    stel_mass = branch.SubhaloMassType[:,4][mask] * 10**10 * (1/Planck15.h)
    gas_mass = branch.SubhaloMassType[:,0][mask] * 10**10 * (1/Planck15.h)
    subhalo_mass = branch.SubhaloMassType[:,1][mask] * 10**10 * (1/Planck15.h)
    BH_mass = branch.SubhaloBHMass[mask] * 10**10 * (1/Planck15.h)
    BH_Mdot = branch.SubhaloBHMdot[mask] * 10**10 * (1/Planck15.h)
    
    # Calculating BH_luminosity. This should deal with single floats or np.ndarray formats.
    log10_Lbh_bol, log10_Lbh_xray = bh_luminosity.compute_luminosity(branch.SubhaloBHMass[mask], branch.SubhaloBHMdot[mask], method=1)
    
    # returning other black hole properties.
    BH_CumEgyInjection_QM, BH_CumEgyInjection_RM, BH_CumMassGrowth_QM, BH_CumMassGrowth_RM, BH_Density, BHpart_count, BH_progenitors = bh_params_subhalo.compute_params_branch(branch.SubfindID[mask], branch.SnapNum[mask], basepath)
	
    # computing cold gas fraction.
    gas_fraction = cold_gas_fraction.compute_fraction_set(branch.SubfindID[mask], branch.SnapNum[mask], basePath=basepath)
	
    # Creating pandas object to output. These are designed to appended to others for other branches.
    tab = pd.DataFrame({'branch_subfind':branch.SubfindID[mask], 'branch_snapnum':branch.SnapNum[mask],
                        'root_subfind':root_sub, 'root_snap':root_snap, 'halo_mass':halo_mass, 'subhalo_mass':subhalo_mass,
                        'central_flag':central_flag, 'stel_mass':stel_mass, 'gas_mass':gas_mass,
                        'cold_gas_fraction':gas_fraction,
                        'BH_mass':BH_mass, 'BH_Mdot':BH_Mdot, 'SFR':branch.SubhaloSFR[mask],
                        'log10_Lbh_bol':log10_Lbh_bol, 'log10_Lbh_xray':log10_Lbh_xray,
                        'BH_CumEgyInjection_QM':BH_CumEgyInjection_QM, 'BH_CumEgyInjection_RM':BH_CumEgyInjection_RM,
                        'BH_CumMassGrowth_QM':BH_CumMassGrowth_QM, 'BH_CumMassGrowth_RM':BH_CumMassGrowth_RM,
                        'BH_local_gas_density':BH_Density, 'BHpart_count':BHpart_count, 'BH_progenitors':BH_progenitors,
                        'GasMetallicity':branch.SubhaloGasMetallicity[mask], 'branch_z':branch_z[mask]})
    return tab


def branch_tabulate_gas_only(subfind, snapnum, tree, lookback_z, basepath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output'):
	'''
	Function which finds the main branch for a given subhalo (subfind_id, snapnum)
	back to a given redshift (lookback_z).

	Returns a pandas dataframe with:
	- total gas fraction within 2Re
	- cold gas fraction within 2Re
	'''
	branch = tree.get_main_branch(snapnum, subfind, keysel=['SubfindID', 'SubhaloMassInRadType', 'SubhaloPos', 'SubhaloSFRinRad', 'SubhaloGasMetallicity', 'SnapNum', 'SubhaloHalfmassRadType'])
	
	# Since branch is constructed from DM only, those halos with zero DM (but stellar comps)
	# may not have ANY branch object.
	if branch == None:
		return pd.DataFrame({})
	
	# Converting snapnums to redshifts
	branch_z = np.array([time_conversions.snap_to_z(i) for i in branch.SnapNum])
	# Creating mask for maximum lookback snapnum.
	mask = (branch_z <= lookback_z)

	# Creating root subfind and snapnum columns.
	root_sub = np.full(branch.SubfindID[mask].shape[0], subfind)
	root_snap = np.full(branch.SnapNum[mask].shape[0], snapnum)

	# Converting masses to consistent units.
	stel_mass_2re = branch.SubhaloMassInRadType[:,4][mask] * 10**10 * (1/Planck15.h)
	gas_mass_2re = branch.SubhaloMassInRadType[:,0][mask] * 10**10 * (1/Planck15.h)

	# computing total gas mass fraction.
	gas_frac_2re = gas_mass_2re / stel_mass_2re

	# computing cold gas fraction.
	cold_gas_mass_2re = cold_gas_fraction.compute_fraction_set(branch.SubfindID[mask], branch.SnapNum[mask], branch.SubhaloHalfmassRadType[:,4][mask], 
															   branch.SubhaloPos[mask], basePath=basepath)
	cold_gas_mass_2re *= 10**10 * (1/Planck15.h)

	# cold gas frac.
	cold_gas_frac_2re = cold_gas_mass_2re / stel_mass_2re

    # Creating pandas object to output. These are designed to appended to others for other branches.
	tab = pd.DataFrame({'branch_subfind':branch.SubfindID[mask], 'branch_snapnum':branch.SnapNum[mask], 
						'root_subfind':root_sub, 'root_snap':root_snap, 
						'stel_mass_2re':stel_mass_2re, 'gas_mass_2re':gas_mass_2re,
						'gas_frac_2re':gas_frac_2re, 'cold_gas_frac_2re':cold_gas_frac_2re, 
						'GasMetallicity_2re':branch.SubhaloGasMetallicity[mask], 'branch_z':branch_z[mask]})
	
	return tab