'''
plot_population - functions to plot different populations in the manga like tng sample.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import split_population
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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

def plot_property_residual(time, property, time_total, total_population_property, ax, label=None, color='k', linestyle='solid'):
    '''
    Given a population of galaxies with a defined property and equivalent redshift, this 
    finds the average value at every snapshot and finds the median and standard error.
    The values for the total population are also found to plot the residual.
    '''
    ts = np.unique(time) 
    # finding average value for each defined z.
    av = np.zeros(ts.shape[0])
    std = np.zeros(ts.shape[0])
    
    av_pop = np.zeros(ts.shape[0])
    std_pop = np.zeros(ts.shape[0])
    
    for i, z in enumerate(ts):
        av[i] = np.mean(property[time == z])
        std[i] = scipy.stats.sem(property[time == z]) 
        av_pop[i] = np.mean(total_population_property[time_total == z])
        std_pop[i] = scipy.stats.sem(total_population_property[time_total == z]) 
        
    ax.plot(ts, av - av_pop, linestyle=linestyle, color=color, label=label)
    ax.fill_between(ts, (av - av_pop) - std, (av - av_pop) + std, alpha=0.2, color=color)
    return

def plot_row_evolution(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
	# Top 33% $M_{stel}$
	#plot_property_evolution(QU_tab_HM.branch_lookback_time.values[QU_tab_HM[property].values > condition], QU_tab_HM[property].values[QU_tab_HM[property].values > condition], ax[0], r'Top 33% $M_{stel}$', color='salmon')
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='salmon', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='salmon', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='salmon', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(QU_tab_LM.branch_lookback_time.values[QU_tab_LM[property].values > condition], QU_tab_LM[property].values[QU_tab_LM[property].values > condition], ax[0], color='steelblue')
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='steelblue', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[0], color='steelblue', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[0], color='steelblue', linestyle='dotted') 
    
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Top 33% $M_{stel}$
	#plot_property_evolution(GV_tab_HM.branch_lookback_time.values[GV_tab_HM[property].values > condition], GV_tab_HM[property].values[GV_tab_HM[property].values > condition], ax[1], color='salmon')
	plot_property_evolution(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(GV_tab_LM.branch_lookback_time.values[GV_tab_LM[property].values > condition], GV_tab_LM[property].values[GV_tab_LM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$', color='steelblue')
	plot_property_evolution(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='steelblue', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition], ax[1], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='steelblue', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='steelblue', linestyle='dotted') 
	# Star forming.
	# Top 33% $M_{stel}$
	#plot_property_evolution(SF_tab_HM.branch_lookback_time.values[SF_tab_HM[property].values > condition], SF_tab_HM[property].values[SF_tab_HM[property].values > condition], ax[2], r'Top 33% $M_{stel}$', color='salmon')
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[2], r'Top 33\% $M_{stel}$, $\Delta$PA > 150',color='salmon', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(SF_tab_LM.branch_lookback_time.values[SF_tab_LM[property].values > condition], SF_tab_LM[property].values[SF_tab_LM[property].values > condition], ax[2], r'Bottom 50\% $M_{stel}$', color='steelblue')
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[2], r'Bottom 33\% $M_{stel}$, $\Delta$PA \geq 150',color='steelblue', linestyle='dotted') 
	return 


def plot_row_residual_LM(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition],
# 						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='steelblue', linestyle='solid') 
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition],
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='steelblue', linestyle='dashed') 
	plot_property_residual(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], 
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$', color='steelblue', linestyle='dotted') 
    
	ax[0].set_xlim([0, 7.85])
	ax[0].invert_xaxis()
	# Green valley.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], 
# 						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='steelblue', linestyle='solid') 
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition],
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='steelblue', linestyle='dashed') 
	plot_property_residual(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], 
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$', color='steelblue', linestyle='dotted') 
	# Star forming.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], 
# 						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='solid') 
	ax[2].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='steelblue', linestyle='dashed') 
	plot_property_residual(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2] ,color='steelblue', linestyle='dotted')  
	return 

def plot_row_residual_HM(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
	# Top 33% $M_{stel}$
# 	plot_property_residual(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition],
# 						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='salmon', linestyle='solid') 
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition],
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='salmon', linestyle='dashed') 
	plot_property_residual(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], 
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='salmon', linestyle='dotted') 
	
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Top 33% $M_{stel}$
# 	plot_property_residual(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition],
# 						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='solid') 
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition],
						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dashed') 
	plot_property_residual(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition],
						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='salmon', linestyle='dotted') 
	# Star forming.
	# Top 33% $M_{stel}$
# 	plot_property_residual(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], 
# 						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='solid') 
	ax[2].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='dashed') 
	plot_property_residual(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='salmon', linestyle='dotted') 
	return 



# ---------------------------------------------------------------------------------------
# func: xtick_format
#
# This function sorts out major and minor tick spacing for nice plots.
# 
# Input: Major and minor tick spacing: (float), ax, format of tick labels.

def xtick_format(major, minor, ax, format='%1.0f'):
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    return 
    
# ---------------------------------------------------------------------------------------
# func: ytick_format
#
# This function sorts out major and minor tick spacing for nice plots.
# 
# Input: Major and minor tick spacing. (float).

def ytick_format(major, minor, ax, format='%1.0f'):
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    return
    