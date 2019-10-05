'''branch properties - returns a variety of properties along the main branch for a given
subhalo'''

import numpy as np
import groupcat as gc 
import h5py
from astropy.cosmology import Planck15, z_at_value
import pandas as pd
import BH_luminosity_funcs as BHlum




# ---------------------------------------------------------------------------------------
# func: branch_tabulate()
# 
# This function finds the main branch for each supplied subfind_id, snapnum combo with a
# cutoff in z (lookback). This returns a pandas object with:
# Mass history (stellar, halo, gas, BH)
# SFR and GasMetallicity.

def branch_tabulate(subfind, snapnum, tree, lookback_z):
    ''' 
    This function finds the main branch for each supplied subfind_id, snapnum combo with a 
    cutoff in z (lookback). This returns a pandas object with:
    Mass history (stellar, halo, gas, BH)
    SFR and GasMetallicity.
    
    New:
    This also now calculates the BH luminosity! Both for bolometric and X-ray.
    Using the simple way with epsilon_r = 0.1
    '''
    
    branch = tree.get_main_branch(snapnum, subfind, keysel=['SubfindID', 'SubhaloMass', 'SubhaloMassType','SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloGr
Nr', 'SubhaloSFR', 'SubhaloGrNr', 'SubhaloGasMetallicity', 'SnapNum'])

    # Since branch is constructed from DM only, those halos with zero DM (but stellar comps)
    # may not have ANY branch object.
    if branch == None:
        return pd.DataFrame({})
 
    # Converting snapnums to redshifts
    branch_z = np.array([snap_to_z(i) for i in branch.SnapNum])
    # Creating mask for maximum lookback snapnum.
    mask = (branch_z <= lookback_z)

    # Finding the halo mass + central/satellite flag.
    halo_mass = np.array([])
    central_flag = np.array([])
    
    # Looping over all subhalos in main branch. Stopping at
    for i in np.arange(branch.SnapNum[mask].shape[0]):
        group_info = gc.loadSingle('/virgo/simulations/IllustrisTNG/L75n1820TNG/output', branch.SnapNum[i], haloID=branch.SubhaloGrNr[i])
        halo_mass = np.append(halo_mass, group_info['GroupMass'])
        # Finding if current subhalo is a central or satellite in current group.
        if branch.SubfindID[i] == group_info['GroupFirstSub']:
            central_flag = np.append(central_flag, int(1)) # central_flag == 1 if central.
        else: 
            central_flag = np.append(central_flag, int(0)) # central_flag == 0 if satellite.
    
    # Creating root subfind and snapnum columns.
    root_sub = np.full(branch.SnapNum[mask].shape[0], subfind)
    root_snap = np.full(branch.SnapNum[mask].shape[0], snapnum)
    
    # Converting masses to consistent units.
    halo_mass = halo_mass * 10**10 * (1/Planck15.h)
    stel_mass = branch.SubhaloMassType[:,4][mask] * 10**10 * (1/Planck15.h)
    gas_mass = branch.SubhaloMassType[:,0][mask] * 10**10 * (1/Planck15.h)
    subhalo_mass = branch.SubhaloMassType[:,1][mask] * 10**10 * (1/Planck15.h)
    BH_mass = branch.SubhaloBHMass[mask] * 10**10 * (1/Planck15.h)
    BH_Mdot = branch.SubhaloBHMdot[mask] * 10**10 * (1/Planck15.h)
    
    # Calculating BH_luminosity. This should deal with single floats or np.ndarray formats.
    log10_Lbh_bol, log10_Lbh_xray = BHlum.compute_luminosity(branch.SubhaloBHMass[mask], branch.SubhaloBHMdot[mask], 1)
        
    # Creating pandas object to output. These are designed to appended to others for other branches.
    tab = pd.DataFrame({'branch_subfind':branch.SubfindID[mask], 'branch_snapnum':branch.SnapNum[mask],
                        'root_subfind':root_sub, 'root_snap':root_snap, 'halo_mass':halo_mass, 'subhalo_mass':subhalo_mass,
                        'central_flag':central_flag, 'stel_mass':stel_mass, 'gas_mass':gas_mass,
                        'BH_mass':BH_mass, 'BH_Mdot':BH_Mdot, 'SFR':branch.SubhaloSFR[mask],
                        'log10_Lbh_bol':log10_Lbh_bol, 'log10_Lbh_xray':log10_Lbh_xray, 
                        'GasMetallicity':branch.SubhaloGasMetallicity[mask], 'branch_z':branch_z[mask]})
    return tab
