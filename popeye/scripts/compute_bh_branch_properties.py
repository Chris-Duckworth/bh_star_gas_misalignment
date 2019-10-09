'''
compute_bh_branch_properties - finds the bh and mass property history.
'''

import numpy as np
import branch_properties
import pandas as pd 
import readtreeHDF5

# ---------------------------------------------------------------------------------------
# Loading in z=0 objects to run script for.

filepath = '/home/cduckworth/bh_star_gas_misalignment/popeye/catalogues/'
basepath = '/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/output/'
treepath = '/simons/scratch/sgenel/IllustrisTNG/L75n1820TNG/postprocessing/trees/SubLink/'

# loading in tree
tree = readtreeHDF5.TreeDB(treepath)
# loading in subfind_ids to consider
tab = pd.read_csv(filepath+'tng100_mpl8_pa_info_v0.1.csv')

# ---------------------------------------------------------------------------------------
# exporting subfind_ids and snapnums.

snapnum = 99

# ---------------------------------------------------------------------------------------
# tetsting for the first 10 subuhalos.

pout = pd.DataFrame({})

for sub in tab.subfind_id.values:
    print(sub)
    pout = pout.append(branch_properties.branch_tabulate(sub, snapnum, tree, 1, basepath))

pout.to_csv(filepath+'tng100_bh_history.csv', index=None)

# ---------------------------------------------------------------------------------------