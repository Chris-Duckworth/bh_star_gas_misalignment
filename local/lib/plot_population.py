'''
plot_population - functions to plot different populations in the manga like tng sample.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import split_population

def plot_property_evolution(time, property, ax, label=None, color='k', linestyle='solid'):
    '''
    Given a population of galaxies with a defined property and equivalent redshift, this 
    finds the average value at every snapshot and finds the median and standard error.
    These are then plotted. 
    '''
    ts = np.unique(time) 
    # finding average value for each defined z.
    av = np.zeros(ts.shape[0])
    std = np.zeros(ts.shape[0])
    for i, z in enumerate(ts):
        av[i] = np.mean(property[time == z])
        std[i] = scipy.stats.sem(property[time == z]) 
    ax.plot(ts, av, linestyle=linestyle, color=color, label=label)
    ax.fill_between(ts, av - std, av + std, alpha=0.2, color=color)
    return


def plot_row_evolution(QU, GV, SF, mass_tab, property, condition, ax):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=33, upper_percentile=66)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=33, upper_percentile=66)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=33, upper_percentile=66)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=33, upper_percentile=66)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=33, upper_percentile=66)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=33, upper_percentile=66)  
	# Quenched.
	# Top 33% $M_{stel}$
	plot_property_evolution(QU_tab_HM.branch_lookback_time.values[QU_tab_HM[property].values > condition], QU_tab_HM[property].values[QU_tab_HM[property].values > condition], ax[0], r'Top 33% $M_{stel}$', color='salmon')
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $< 30$', color='salmon', linestyle='dotted') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $> 30$',color='salmon', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $> 150$',color='salmon', linestyle='-.') 
	# Bottom 33% $M_{stel}$
	plot_property_evolution(QU_tab_LM.branch_lookback_time.values[QU_tab_LM[property].values > condition], QU_tab_LM[property].values[QU_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$', color='steelblue')
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$, $\Delta$PA $< 30$', color='steelblue', linestyle='dotted') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$, $\Delta$PA $> 30$',color='steelblue', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$, $\Delta$PA $> 150$',color='steelblue', linestyle='-.') 

	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Top 33% $M_{stel}$
	plot_property_evolution(GV_tab_HM.branch_lookback_time.values[GV_tab_HM[property].values > condition], GV_tab_HM[property].values[GV_tab_HM[property].values > condition], ax[1], r'Top 33% $M_{stel}$', color='salmon')
	plot_property_evolution(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dotted') 
	plot_property_evolution(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$, $\Delta$PA > 150',color='salmon', linestyle='-.') 
	# Bottom 33% $M_{stel}$
	plot_property_evolution(GV_tab_LM.branch_lookback_time.values[GV_tab_LM[property].values > condition], GV_tab_LM[property].values[GV_tab_LM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$', color='steelblue')
	plot_property_evolution(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$, $\Delta$PA $< 30$', color='steelblue', linestyle='dotted') 
	plot_property_evolution(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition], ax[1], 'Bottom 33% $M_{stel}$, $\Delta$PA $> 30$',color='steelblue', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$, $\Delta$PA $> 150$',color='steelblue', linestyle='-.') 
	# Star forming.
	# Top 33% $M_{stel}$
	plot_property_evolution(SF_tab_HM.branch_lookback_time.values[SF_tab_HM[property].values > condition], SF_tab_HM[property].values[SF_tab_HM[property].values > condition], ax[2], r'Top 33% $M_{stel}$', color='salmon')
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='dotted') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[2], r'Top 33\% $M_{stel}$, $\Delta$PA > 150',color='salmon', linestyle='-.') 
	# Bottom 33% $M_{stel}$
	plot_property_evolution(SF_tab_LM.branch_lookback_time.values[SF_tab_LM[property].values > condition], SF_tab_LM[property].values[SF_tab_LM[property].values > condition], ax[2], r'Bottom 50\% $M_{stel}$', color='steelblue')
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='dotted') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[2], r'Bottom 33\% $M_{stel}$, $\Delta$PA > 150',color='steelblue', linestyle='-.') 
	return 
