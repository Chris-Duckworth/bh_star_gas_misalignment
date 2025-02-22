'''
plot_population - functions to plot different populations in the manga like tng sample.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import split_population
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plot_property_evolution(time, property, ax, label=None, color='k', linestyle='solid', alpha=0.3):
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
    ax.plot(ts, av, linestyle=linestyle, color=color, label=label, linewidth=3)
    ax.fill_between(ts, av - std, av + std, alpha=alpha, color=color)
    return

def plot_property_residual(time, property, time_total, total_population_property, ax, label=None, color='k', linestyle='solid', alpha=0.2, peak=False):
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
    
    resids = av - av_pop
    ax.plot(ts, resids, linestyle=linestyle, color=color, label=label, linewidth=3)
    ax.fill_between(ts, resids - std, resids + std, alpha=alpha, color=color)
    
    if peak == True:
        # this finds the midpoint of energy injection for the residuals.
        total_resid = np.sum(resids)
        for ind, val in enumerate(np.cumsum(resids)):
            if val > total_resid / 2:
                # finding point at which half of total energy is injected.
                ax.axvline(ts[ind], color=color, linewidth=5, alpha=0.5)
                break
    return

def plot_row_evolution(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
	# Top 33% $M_{stel}$
	#plot_property_evolution(QU_tab_HM.branch_lookback_time.values[QU_tab_HM[property].values > condition], QU_tab_HM[property].values[QU_tab_HM[property].values > condition], ax[0], r'Top 33% $M_{stel}$', color='slategrey')
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='slategrey', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='slategrey', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='slategrey', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(QU_tab_LM.branch_lookback_time.values[QU_tab_LM[property].values > condition], QU_tab_LM[property].values[QU_tab_LM[property].values > condition], ax[0], color='darkseagreen')
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dotted') 
    
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Top 33% $M_{stel}$
	#plot_property_evolution(GV_tab_HM.branch_lookback_time.values[GV_tab_HM[property].values > condition], GV_tab_HM[property].values[GV_tab_HM[property].values > condition], ax[1], color='slategrey')
	plot_property_evolution(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(GV_tab_LM.branch_lookback_time.values[GV_tab_LM[property].values > condition], GV_tab_LM[property].values[GV_tab_LM[property].values > condition], ax[1], r'Bottom 33% $M_{stel}$', color='darkseagreen')
	plot_property_evolution(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='darkseagreen', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition], ax[1], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='darkseagreen', linestyle='dotted') 
	# Star forming.
	# Top 33% $M_{stel}$
	#plot_property_evolution(SF_tab_HM.branch_lookback_time.values[SF_tab_HM[property].values > condition], SF_tab_HM[property].values[SF_tab_HM[property].values > condition], ax[2], r'Top 33% $M_{stel}$', color='slategrey')
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[2], r'Top 33\% $M_{stel}$, $\Delta$PA > 150',color='slategrey', linestyle='dotted') 
	# Bottom 33% $M_{stel}$
	#plot_property_evolution(SF_tab_LM.branch_lookback_time.values[SF_tab_LM[property].values > condition], SF_tab_LM[property].values[SF_tab_LM[property].values > condition], ax[2], r'Bottom 50\% $M_{stel}$', color='darkseagreen')
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[2], r'Bottom 33\% $M_{stel}$, $\Delta$PA \geq 150',color='darkseagreen', linestyle='dotted') 
	return 

def plot_row_evolution_BHmass(QU, GV, SF, mass_tab, property, condition, ax, lower_mass=10**8, upper_mass=10**8):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	
	This splits based on absolute value of mass rather than percentile.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_BHmass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	# Quenched.
	# Kinetic mass range
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'$M_{BH} \geq 10^{8}M_{\odot}$, $\Delta$PA $< 30^{\circ}$', color='slategrey', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[0], r'$M_{BH} \geq 10^{8}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='slategrey', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[0], r'$M_{BH} \geq 10^{8}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$',color='slategrey', linestyle='dotted') 
	# Quasar mass range
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dotted') 
    
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Kinetic mass range
	plot_property_evolution(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dotted') 
	# Quasar mass range
	plot_property_evolution(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'$M_{BH} < 10^{8}M_{\odot}$, $\Delta$PA $< 30^{\circ}$', color='darkseagreen', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition], ax[1], '$M_{BH} < 10^{8}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], ax[1], r'$M_{BH} < 10^{8}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$',color='darkseagreen', linestyle='dotted') 
	# Star forming.
	# Kinetic mass range
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[2], r'Top 33\% $M_{stel}$, $\Delta$PA > 150',color='slategrey', linestyle='dotted') 
	# Quasar mass range
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[2], r'Bottom 33\% $M_{stel}$, $\Delta$PA \geq 150',color='darkseagreen', linestyle='dotted') 
	return 

def plot_row_evolution_mass(QU, GV, SF, mass_tab, property, condition, ax, lower_mass=10**10.2, upper_mass=10**10.2):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	
	This splits based on absolute value of mass rather than percentile.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	# Quenched.
	# Kinetic mass range
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $< 30^{\circ}$', color='slategrey', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[0], r'$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='slategrey', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[0], r'$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$',color='slategrey', linestyle='dotted') 
	# Quasar mass range
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='dotted', alpha=0.1) 
    
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Kinetic mass range
	plot_property_evolution(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dotted', alpha=0.1)
	# Quasar mass range
	plot_property_evolution(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $< 30^{\circ}$', color='darkseagreen', linestyle='solid') 
	plot_property_evolution(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition], ax[1], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], ax[1], r'$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$',color='darkseagreen', linestyle='dotted', alpha=0.1)
	# Star forming.
	# Kinetic mass range
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[2], r'Top 33\% $M_{stel}$, $\Delta$PA > 150',color='slategrey', linestyle='dotted', alpha=0.1)
	# Quasar mass range
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[2], r'Bottom 33\% $M_{stel}$, $\Delta$PA \geq 150',color='darkseagreen', linestyle='dotted', alpha=0.1)
	return 


def plot_two_evolution_mass(QU, SF, mass_tab, property, condition, ax, lower_mass=10**10.2, upper_mass=10**10.2):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	
	This splits based on absolute value of mass rather than percentile.
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)

	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	
	# Star forming.
	# Kinetic mass range
	plot_property_evolution(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[0], color='slategrey', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], ax[0], color='slategrey', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], ax[0], color='slategrey', linestyle='dotted', alpha=0.1)
	# Quasar mass range
	plot_property_evolution(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[0], '$\mathrm{M_{stel} < 10^{10.2}M_{\odot}, \Delta PA < 30^{\circ} }$', color='darkseagreen', linestyle='solid') 
	plot_property_evolution(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], ax[0], '$\mathrm{M_{stel} < 10^{10.2}M_{\odot}, \Delta PA \geq 30^{\circ} }$', color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], ax[0], '$\mathrm{M_{stel} < 10^{10.2}M_{\odot}, \Delta PA \geq 150^{\circ} }$', color='darkseagreen', linestyle='dotted', alpha=0.1)
	
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	
	# Quenched.
	# Kinetic mass range
	plot_property_evolution(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[1], '$\mathrm{M_{stel} \geq 10^{10.2}M_{\odot}, \Delta PA < 30^{\circ} }$', color='slategrey', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition], ax[1], '$\mathrm{M_{stel} \geq 10^{10.2}M_{\odot}, \Delta PA \geq 30^{\circ} }$',color='slategrey', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], ax[1], '$\mathrm{M_{stel} \geq 10^{10.2}M_{\odot}, \Delta PA \geq 150^{\circ} }$',color='slategrey', linestyle='dotted') 
	# Quasar mass range
	plot_property_evolution(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[1], color='darkseagreen', linestyle='solid') 
	plot_property_evolution(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition], ax[1], color='darkseagreen', linestyle='dashed') 
	plot_property_evolution(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], ax[1], color='darkseagreen', linestyle='dotted', alpha=0.1) 
    
	return 


def plot_row_residual_LM_percentile(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66, peak=False):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition],
# 						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], color='darkseagreen', linestyle='solid') 
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition],
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$', color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], 
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1, peak=peak)
    
	ax[0].set_xlim([0, 7.85])
	ax[0].invert_xaxis()
	# Green valley.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], 
# 						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='darkseagreen', linestyle='solid') 
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition],
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], 'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], 
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'Bottom 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1, peak=peak)
	# Star forming.
 	# Bottom 33% $M_{stel}$
# 	plot_property_residual(SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], 
# 						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='solid') 
	ax[2].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2] ,color='darkseagreen', linestyle='dotted', alpha=0.1, peak=peak)
	return 

def plot_row_residual_HM_percentile(QU, GV, SF, mass_tab, property, condition, ax, lower_percentile=33, upper_percentile=66):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and percentiles of mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(QU, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(GV, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=30, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass_percentile(SF, mass_tab, lower_PA=30, upper_PA=150, lower_percentile=lower_percentile, upper_percentile=upper_percentile)  
	# Quenched.
	# Top 33% $M_{stel}$
# 	plot_property_residual(QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition],
# 						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $< 30^{\circ}$', color='slategrey', linestyle='solid') 
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition],
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 30^{\circ}$',color='slategrey', linestyle='dashed') 
	plot_property_residual(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], 
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[0], r'Top 33\% $M_{stel}$, $\Delta$PA $\geq 150^{\circ}$',color='slategrey', linestyle='dotted') 
	
	ax[0].set_xlim([0, 7.93])
	ax[0].invert_xaxis()
	# Green valley.
	# Top 33% $M_{stel}$
# 	plot_property_residual(GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition],
# 						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='solid') 
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(GV_mis_tab_HM.branch_lookback_time.values[GV_mis_tab_HM[property].values > condition], GV_mis_tab_HM[property].values[GV_mis_tab_HM[property].values > condition],
						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dashed') 
	plot_property_residual(GV_counter_tab_HM.branch_lookback_time.values[GV_counter_tab_HM[property].values > condition], GV_counter_tab_HM[property].values[GV_counter_tab_HM[property].values > condition],
						   GV_align_tab_HM.branch_lookback_time.values[GV_align_tab_HM[property].values > condition], GV_align_tab_HM[property].values[GV_align_tab_HM[property].values > condition], ax[1], color='slategrey', linestyle='dotted') 
	# Star forming.
	# Top 33% $M_{stel}$
# 	plot_property_residual(SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], 
# 						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='solid') 
	ax[2].axhline(0, linestyle='dashed', color='k', alpha=0.3)
	plot_property_residual(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='dashed') 
	plot_property_residual(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[2], color='slategrey', linestyle='dotted') 
	return 

def plot_row_residual_LM(QU, GV, SF, mass_tab, property, condition, ax, lower_mass=10**10.2, upper_mass=10**10.2, peak=False):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and stellar mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	GV_tab_HM, GV_align_tab_HM, GV_mis_tab_HM, GV_tab_LM, GV_align_tab_LM, GV_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, GV_counter_tab_HM, _, _, GV_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(GV, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	
	# Quenched.
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition],
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], 
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[0], r'$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1)
    
	ax[0].set_xlim([0, 7.85])
	ax[0].invert_xaxis()
	# Green valley.
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(GV_mis_tab_LM.branch_lookback_time.values[GV_mis_tab_LM[property].values > condition], GV_mis_tab_LM[property].values[GV_mis_tab_LM[property].values > condition],
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(GV_counter_tab_LM.branch_lookback_time.values[GV_counter_tab_LM[property].values > condition], GV_counter_tab_LM[property].values[GV_counter_tab_LM[property].values > condition], 
						   GV_align_tab_LM.branch_lookback_time.values[GV_align_tab_LM[property].values > condition], GV_align_tab_LM[property].values[GV_align_tab_LM[property].values > condition], ax[1], r'$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1)
	# Star forming.
	ax[2].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2], color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[2] ,color='darkseagreen', linestyle='dotted', alpha=0.1)
	return 

def plot_two_residual_LM(QU, SF, mass_tab, property, condition, ax, lower_mass=10**10.2, upper_mass=10**10.2, peak=False):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and stellar mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	
	# Star forming.
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(SF_mis_tab_LM.branch_lookback_time.values[SF_mis_tab_LM[property].values > condition], SF_mis_tab_LM[property].values[SF_mis_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[0], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$', color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(SF_counter_tab_LM.branch_lookback_time.values[SF_counter_tab_LM[property].values > condition], SF_counter_tab_LM[property].values[SF_counter_tab_LM[property].values > condition], 
						   SF_align_tab_LM.branch_lookback_time.values[SF_align_tab_LM[property].values > condition], SF_align_tab_LM[property].values[SF_align_tab_LM[property].values > condition], ax[0], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1)
	
	ax[0].set_xlim([0, 7.85])
	ax[0].invert_xaxis()
	
	# Quenched.
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(QU_mis_tab_LM.branch_lookback_time.values[QU_mis_tab_LM[property].values > condition], QU_mis_tab_LM[property].values[QU_mis_tab_LM[property].values > condition],
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[1], '$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='darkseagreen', linestyle='dashed', peak=peak)
	plot_property_residual(QU_counter_tab_LM.branch_lookback_time.values[QU_counter_tab_LM[property].values > condition], QU_counter_tab_LM[property].values[QU_counter_tab_LM[property].values > condition], 
						   QU_align_tab_LM.branch_lookback_time.values[QU_align_tab_LM[property].values > condition], QU_align_tab_LM[property].values[QU_align_tab_LM[property].values > condition], ax[1], r'$M_{stel} < 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='darkseagreen', linestyle='dotted', alpha=0.1)

	return 


def plot_two_residual_HM(QU, SF, mass_tab, property, condition, ax, lower_mass=10**10.2, upper_mass=10**10.2, peak=False):
	'''
	Given a property (and a minimum value cut on the property), this returns a full split
	on both morphology and stellar mass. The axes object supplied should be 1 x 3.
	This plots the residuals with respect to the total population! 
	'''
	# Splitting table data.
	QU_tab_HM, QU_align_tab_HM, QU_mis_tab_HM, QU_tab_LM, QU_align_tab_LM, QU_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)
	_, _, QU_counter_tab_HM, _, _, QU_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(QU, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)
	
	SF_tab_HM, SF_align_tab_HM, SF_mis_tab_HM, SF_tab_LM, SF_align_tab_LM, SF_mis_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=30, lower_mass=lower_mass, upper_mass=upper_mass)  
	_, _, SF_counter_tab_HM, _, _, SF_counter_tab_LM = split_population.combine_with_tree_split_on_pa_and_mass(SF, mass_tab, lower_PA=30, upper_PA=150, lower_mass=lower_mass, upper_mass=upper_mass)  
	
	# Star forming.
	ax[0].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(SF_mis_tab_HM.branch_lookback_time.values[SF_mis_tab_HM[property].values > condition], SF_mis_tab_HM[property].values[SF_mis_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[0], '$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$', color='slategrey', linestyle='dashed', peak=peak)
	plot_property_residual(SF_counter_tab_HM.branch_lookback_time.values[SF_counter_tab_HM[property].values > condition], SF_counter_tab_HM[property].values[SF_counter_tab_HM[property].values > condition], 
						   SF_align_tab_HM.branch_lookback_time.values[SF_align_tab_HM[property].values > condition], SF_align_tab_HM[property].values[SF_align_tab_HM[property].values > condition], ax[0], '$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='slategrey', linestyle='dotted', alpha=0.1)
	
	ax[0].set_xlim([0, 7.85])
	ax[0].invert_xaxis()
	
	# Quenched.
	ax[1].axhline(0, linestyle='dashed', color='k', alpha=0.3, linewidth=3)
	plot_property_residual(QU_mis_tab_HM.branch_lookback_time.values[QU_mis_tab_HM[property].values > condition], QU_mis_tab_HM[property].values[QU_mis_tab_HM[property].values > condition],
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[1], '$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 30^{\circ}$',color='slategrey', linestyle='dashed', peak=peak)
	plot_property_residual(QU_counter_tab_HM.branch_lookback_time.values[QU_counter_tab_HM[property].values > condition], QU_counter_tab_HM[property].values[QU_counter_tab_HM[property].values > condition], 
						   QU_align_tab_HM.branch_lookback_time.values[QU_align_tab_HM[property].values > condition], QU_align_tab_HM[property].values[QU_align_tab_HM[property].values > condition], ax[1], r'$M_{stel} \geq 10^{10.2}M_{\odot}$, $\Delta$PA $\geq 150^{\circ}$', color='slategrey', linestyle='dotted', alpha=0.1)
	return 


def xtick_format(major, minor, ax, format='%1.0f'):
    '''
    wrapper function for minor/major x tick formatting.
    '''
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    return 

def ytick_format(major, minor, ax, format='%1.0f'):
    '''
    wrapper function for minor/major y tick formatting.
    '''
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    return
