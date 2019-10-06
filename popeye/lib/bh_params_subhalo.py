'''
bh_params_subhalo - finds additional parameters about the black hole not automatically
provided in subhalo catalogues. basically just summing individual particle info.
'''

import numpy as np
import snapshot as ss

def compute_params(subfind_id, snapnum, basePath='/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output'):
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

       Returns (naming convention follows individual BH particles in TNG)
       -------
       BH_CumEgyInjection_QM : float (units: Msol / h (ckpc/h)^2 / (0.978Gyr/h)^2 )
           Cumulative amount of thermal AGN feedback energy injected into surrounding gas
           in the high-accretion-state (quasar) mode, total over the entire lifetime of
           all blackhole particles in this subhalo.

       BH_CumEgyInjection_RM : float (units: Msol / h (ckpc/h)^2 / (0.978Gyr/h)^2 )
           Cumulative amount of kinetic AGN feedback energy injected into surrounding gas
           in the low-accretion-state (wind) mode, total over the entire lifetime of
           all blackhole particles in this subhalo.

       BH_CumMassGrowth_QM : float (units: Msol / h)
           Total mass accreted onto the BH while in the high-accretion-state (quasar)
           mode, total over the entire lifetime of all blackhole particles in this subhalo.

       BH_CumMassGrowth_RM : float (units: Msol / h)
           Total mass accreted onto the BH while in the low-accretion-state (kinetic wind)
           mode, total over the entire lifetime of all blackhole particles in this subhalo.

       BH_Density : float (units: Msol/h / (ckpc/h)^3)
           Local comoving gas density averaged over the nearest neighbors of the BH.
           Returns the mean value for all BH particles.

       count : float/int
           Number of black hole particles in the subhalo

       total_progenitors : float/int
           Total number of black hole progenitor particles (BH_Progs summed over all
           BH particles.)
    '''

    # loading in all black hole particles in this subhalo.
    props = ss.loadSubhalo(basePath=basePath, snapNum=snapnum, id=subfind_id, partType='BH', fields = ['BH_CumEgyInjection_QM', 'BH_CumEgyInjection_RM', 'BH_CumMassGrowth_QM', 'BH_CumMassGrowth_RM', 'BH_Density', 'BH_Progs'])

    if props['count'] == 0:
        # If no black hole in the subhalo returning -inf for all values.
        return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0,
    elif props['count'] == 1:
        return 1e10*props['BH_CumEgyInjection_QM'][0], 1e10*props['BH_CumEgyInjection_RM'][0], 1e10*props['BH_CumMassGrowth_QM'][0], 1e10*props['BH_CumMassGrowth_RM'][0], 1e10*props['BH_Density'][0], props['count'], props['BH_Progs'][0]
    else:
        # finding summations and averages.
        return 1e10*np.sum(props['BH_CumEgyInjection_QM']), 1e10*np.sum(props['BH_CumEgyInjection_RM']), 1e10*np.sum(props['BH_CumMassGrowth_QM']), 1e10*np.sum(props['BH_CumMassGrowth_RM']), np.mean(1e10*props['BH_Density']), props['count'], np.sum(props['BH_Progs'])
