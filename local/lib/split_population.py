'''
split_populations - functions to split the tng100 manga like sample.
'''

import numpy as np
import pandas as pd


def tng100_pa_sample(tng100_main):
    '''
    For the matched manga-like TNG100 sample, this returns the pa defined. Since our TNG100 
    sample is already only matched to main sample manga galaxies we dont have to do the pre-
    selection.
    '''
    tng100_pa = tng100_main[(tng100_main.stel_feature == 0) & (tng100_main.halpha_feature == 0) & ((tng100_main.stel_qual == 1) | (tng100_main.stel_qual == 2)) & ((tng100_main.halpha_qual == 1) | (tng100_main.halpha_qual == 2))]
    return tng100_pa


def SFMS_breakdown(tab):
    '''
    This function splits into individual dataframes based on the flags in annalisa's flags.
    '''
    QU = tab[tab.sfms_flag == 0]
    SF = tab[tab.sfms_flag == 1]
    GV = tab[tab.sfms_flag == -1]
    return QU, SF, GV


def combine_with_tree_split_on_pa(tab, tree_tab, lower_PA=30, upper_PA=30):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on PA. Returns 3 
    tables; total info, aligned and misaligned
    '''
    all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values)]
    align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.pa_offset.values < lower_PA])]
    mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.pa_offset.values >= upper_PA])]
    
    print('All:'+str(all_tab.shape[0] / 50)+' Aligned:'+str(align_tab.shape[0] /50)+' Misaligned:'+str(mis_tab.shape[0] /50))

    return all_tab, align_tab, mis_tab

def combine_with_tree_split_on_BHlum(tab, tree_tab, BHlum=44):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on BH luminosity at 
    z=0. Does not return tree info - only for z=0 to plot PA distributions!!
    '''
    all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values)]
    z0_all_tab = all_tab[all_tab.branch_snapnum.values == all_tab.root_snap.values]
    z0_all_tab = tab.merge(z0_all_tab, left_on='subfind_id', right_on='branch_subfind')
    
    high_lum = z0_all_tab[z0_all_tab.log10_Lbh_bol.values >= BHlum]
    low_lum = z0_all_tab[z0_all_tab.log10_Lbh_bol.values < BHlum]
    return z0_all_tab, low_lum, high_lum
    
def combine_with_tree_split_on_BHlum_percentile(tab, tree_tab, lower_percentile=33, upper_percentile=66):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on percentiles of BH 
    luminosity at z=0. Does not return tree info - only for z=0 to plot PA distributions!!
    '''
    all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values)]
    z0_all_tab = all_tab[all_tab.branch_snapnum.values == all_tab.root_snap.values]
    z0_all_tab = tab.merge(z0_all_tab, left_on='subfind_id', right_on='branch_subfind')

    lower_BHlum = np.percentile(z0_all_tab.log10_Lbh_bol.values, lower_percentile)
    upper_BHlum = np.percentile(z0_all_tab.log10_Lbh_bol.values, upper_percentile)
	
    high_lum = z0_all_tab[z0_all_tab.log10_Lbh_bol.values >= upper_BHlum]
    low_lum = z0_all_tab[z0_all_tab.log10_Lbh_bol.values < lower_BHlum]
    return z0_all_tab, low_lum, high_lum

def combine_with_tree_split_on_pa_and_group(tab, tree_tab, lower_PA=30, upper_PA=30):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on both PA and group
    membership (i.e. central or satellite). Returns 6 tables; total info, aligned and misaligned
    for centrals and then satellites.
    '''
    cen_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.central_flag.values == 1])]
    cen_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.central_flag.values == 1)])]
    cen_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.central_flag.values == 1)] )]
    
    print('Centrals. All:'+str(cen_all_tab.shape[0] / 50)+' Aligned:'+str(cen_align_tab.shape[0] /50)+' Misaligned:'+str(cen_mis_tab.shape[0] /50))
    
    sat_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.central_flag.values == 0])]
    sat_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.central_flag.values == 0)])]
    sat_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.central_flag.values == 0)] )]
    
    print('Satellites. All:'+str(sat_all_tab.shape[0] / 50)+' Aligned:'+str(sat_align_tab.shape[0] /50)+' Misaligned:'+str(sat_mis_tab.shape[0] /50))
	
    return cen_all_tab, cen_align_tab, cen_mis_tab, sat_all_tab, sat_align_tab, sat_mis_tab


def combine_with_tree_split_on_pa_and_mass_percentile(tab, tree_tab, lower_PA=30, upper_PA=30, lower_percentile=25, upper_percentile=75, verbose=False):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on both PA and mass 
    of the object at z=0.
    By default the top and bottom quartile are returned.
    Returns 6 tables; total info, aligned and misaligned for high mass and low mass.
    ''' 
    lower_mass = np.percentile(tab.stel_mass.values, lower_percentile)
    upper_mass = np.percentile(tab.stel_mass.values, upper_percentile)
	
    high_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.stel_mass.values > upper_mass])]
    high_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.stel_mass.values > upper_mass)])]
    high_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.stel_mass.values > upper_mass)] )]
        
    low_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.stel_mass.values <= lower_mass])]
    low_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.stel_mass.values <= lower_mass)])]
    low_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.stel_mass.values <= lower_mass)] )]
    
    if verbose == True:
        print('High mass. All:'+str(high_mass_all_tab.shape[0] / 50)+' Aligned:'+str(high_mass_align_tab.shape[0] /50)+' Misaligned:'+str(high_mass_mis_tab.shape[0] /50))
        print('Low mass.  All:'+str(low_mass_all_tab.shape[0] /50)+' Aligned:'+str(low_mass_align_tab.shape[0] / 50)+' Misaligned:'+str(low_mass_mis_tab.shape[0] / 50))
    
    return high_mass_all_tab, high_mass_align_tab, high_mass_mis_tab, low_mass_all_tab, low_mass_align_tab, low_mass_mis_tab

def combine_with_tree_split_on_pa_and_mass(tab, tree_tab, lower_PA=30, upper_PA=30, lower_mass=10**10, upper_mass=10**11, verbose=False):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on both PA and mass 
    of the object at z=0.
    By default the top and bottom quartile are returned.
    Returns 6 tables; total info, aligned and misaligned for high mass and low mass.
    ''' 

    high_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.stel_mass.values > upper_mass])]
    high_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.stel_mass.values > upper_mass)])]
    high_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.stel_mass.values > upper_mass)] )]
        
    low_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.stel_mass.values <= lower_mass])]
    low_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.stel_mass.values <= lower_mass)])]
    low_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.stel_mass.values <= lower_mass)] )]
    
    if verbose == True:
        print('High mass. All:'+str(high_mass_all_tab.shape[0] / 50)+' Aligned:'+str(high_mass_align_tab.shape[0] /50)+' Misaligned:'+str(high_mass_mis_tab.shape[0] /50))
        print('Low mass.  All:'+str(low_mass_all_tab.shape[0] /50)+' Aligned:'+str(low_mass_align_tab.shape[0] / 50)+' Misaligned:'+str(low_mass_mis_tab.shape[0] / 50))
    
    return high_mass_all_tab, high_mass_align_tab, high_mass_mis_tab, low_mass_all_tab, low_mass_align_tab, low_mass_mis_tab

def combine_with_tree_split_on_pa_and_BHmass(tab, tree_tab, lower_PA=30, upper_PA=30, lower_mass=10**8, upper_mass=10**8, verbose=False):
    '''
    Function that takes a defined sample in the TNG100 - MPL-8 matched population, combines
    with supplementary tree info (for the main branch) and then splits on both PA and black 
    hole mass of the object at z=0.
    Returns 6 tables; total info, aligned and misaligned for high mass and low mass.
    ''' 

    high_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.BHmass.values > upper_mass])]
    high_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.BHmass.values > upper_mass)])]
    high_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.BHmass.values > upper_mass)] )]
        
    low_mass_all_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[tab.BHmass.values <= lower_mass])]
    low_mass_align_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values < lower_PA) & (tab.BHmass.values <= lower_mass)])]
    low_mass_mis_tab = tree_tab[tree_tab.root_subfind.isin(tab.subfind_id.values[(tab.pa_offset.values >= upper_PA) & (tab.BHmass.values <= lower_mass)] )]
    
    if verbose == True:
        print('High mass. All:'+str(high_mass_all_tab.shape[0] / 50)+' Aligned:'+str(high_mass_align_tab.shape[0] /50)+' Misaligned:'+str(high_mass_mis_tab.shape[0] /50))
        print('Low mass.  All:'+str(low_mass_all_tab.shape[0] /50)+' Aligned:'+str(low_mass_align_tab.shape[0] / 50)+' Misaligned:'+str(low_mass_mis_tab.shape[0] / 50))
    
    return high_mass_all_tab, high_mass_align_tab, high_mass_mis_tab, low_mass_all_tab, low_mass_align_tab, low_mass_mis_tab


def snap_to_z(snapnum):
    dict = {0: 20.046490988807516, 1: 14.989173240042412, 2: 11.980213315300293,
            3: 10.975643294137885, 4: 9.996590466186333, 5: 9.388771271940549, 
            6: 9.00233985416247, 7: 8.449476294368743, 8: 8.012172948865935, 
            9: 7.5951071498715965, 10: 7.23627606616736, 11: 7.005417045544533, 
            12: 6.491597745667503, 13: 6.0107573988449, 14: 5.846613747881867, 
            15: 5.5297658079491026, 16: 5.227580973127337, 17: 4.995933468164624, 
            18: 4.664517702470927, 19: 4.428033736605549, 20: 4.176834914726472, 
            21: 4.0079451114652676, 22: 3.7087742646422353, 23: 3.4908613692606485, 
            24: 3.2830330579565246, 25: 3.008131071630377, 26: 2.8957850057274284, 
            27: 2.7331426173187188, 28: 2.5772902716018935, 29: 2.4442257045541464, 
            30: 2.3161107439568918, 31: 2.207925472383703, 32: 2.1032696525957713, 
            33: 2.0020281392528516, 34: 1.9040895435327672, 35: 1.822689252620354, 
            36: 1.7435705743308647, 37: 1.6666695561144653, 38: 1.6042345220731056, 
            39: 1.5312390291576135, 40: 1.4955121664955557, 41: 1.4140982203725216, 
            42: 1.3575766674029972, 43: 1.3023784599059653, 44: 1.2484726142451428, 
            45: 1.2062580807810006, 46: 1.1546027123602154, 47: 1.1141505637653806, 
            48: 1.074457894547674, 49: 1.035510445664141, 50: 0.9972942257819404, 
            51: 0.9505313515850327, 52: 0.9230008161779089, 53: 0.8868969375752482, 
            54: 0.8514709006246495, 55: 0.8167099790118506, 56: 0.7910682489463392, 
            57: 0.7574413726158526, 58: 0.7326361820223115, 59: 0.7001063537185233, 
            60: 0.6761104112134777, 61: 0.6446418406845371, 62: 0.6214287452425136, 
            63: 0.5985432881875667, 64: 0.5759808451078874, 65: 0.5463921831410221, 
            66: 0.524565820433923, 67: 0.5030475232448832, 68: 0.4818329434209512, 
            69: 0.4609177941806475, 70: 0.4402978492477432, 71: 0.41996894199726653, 
            72: 0.3999269646135635, 73: 0.38016786726023866, 74: 0.36068765726181673, 
            75: 0.3478538418581776, 76: 0.32882972420595435, 77: 0.31007412012783386, 
            78: 0.2977176845174465, 79: 0.2733533465784399, 80: 0.2613432561610123, 
            81: 0.24354018155467028, 82: 0.22598838626019768, 83: 0.21442503551449454, 
            84: 0.19728418237600986, 85: 0.1803852617057493, 86: 0.1692520332436107, 
            87: 0.15274876890238098, 88: 0.14187620396956202, 89: 0.12575933241126092, 
            90: 0.10986994045882548, 91: 0.09940180263022191, 92: 0.08388443079747931, 
            93: 0.07366138465643868, 94: 0.058507322794512984, 95: 0.04852362998180593, 
            96: 0.0337243718735154, 97: 0.023974428382762536, 98: 0.009521666967944764, 
            99: 2.220446049250313e-16}
    return dict[snapnum]