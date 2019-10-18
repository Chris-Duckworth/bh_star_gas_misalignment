'''
plot_population - functions to plot different populations in the manga like tng sample.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def plot_property_evolution(branch_z, property, ax, label=None, color='k', linestyle='solid'):
    '''
    Given a population of galaxies with a defined property and equivalent redshift, this 
    finds the average value at every snapshot and finds the median and standard error.
    These are then plotted. 
    '''
    ts = np.unique(branch_z) 
    # finding average value for each defined z.
    av = np.zeros(ts.shape[0])
    std = np.zeros(ts.shape[0])
    for i, z in enumerate(ts):
        av[i] = np.median(property[branch_z == z])
        std[i] = scipy.stats.sem(property[branch_z == z]) 
    ax.plot(ts, av, linestyle=linestyle, color=color, label=label)
    ax.fill_between(ts, av - std, av + std, alpha=0.2, color=color)
    return
    