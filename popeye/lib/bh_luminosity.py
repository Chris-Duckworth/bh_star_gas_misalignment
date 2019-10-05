'''
bh_luminosity - find the black hole luminosity for a given subhalo/halo
code courtesy of Mélanie Habouzit
'''

import numpy as np
from astropy.cosmology import Planck15

h = Planck15.h
cte_G = 6.67408*10**-8 ## cm3 g-1 s-2                                                                                                                                               
cte_m_p = 1.6726*10**-24 ## g                                                                                                                                                       
cte_eps_r = 0.1
cte_sigma_t = 6.6524*10**-25 ## cm2                                                                                                                                                 
cte_c = 2.998*10**10 ## cm s-1 

def compute_luminosity_MH(mbh, accbh, radia_efficient_agn):
    '''This function calculates the BH luminosity through one of two methods. Required
       inputs are black hole mass (need for eddington fraction calculation) and instantaneous
       accretion rate onto the black hole. The third input dictates the method.
       
       radia_efficient_agn=1 for the easy method to compute luminosity 
       radia_efficient_agn=2 for a more sophisticated model including radiatively inefficient agn
       cte_eps_r=0.1 or (0.2) is the radiative efficiency
       
       This version is directly taken from Melanie Habouzit. Only use for list/array like.
       
       For more information see: Habouzit+19: Linking galaxy structural properties... 
       '''

    mbh= np.array([p*1e10/h for p in mbh]) ## Msun                                                                                                                                    
    accbh= np.array([p*10.22 for p in accbh]) ## Msun/yr                                                                                                                              
    edd = np.array([4*np.pi*cte_G*cte_m_p*p/cte_eps_r/cte_sigma_t/cte_c*(3.154*10**7) for p in mbh])
    fedd = np.array([np.log10(accbh[p]/edd[p]) for p in range(len(edd))])
    
    ## easy method
    if radia_efficient_agn==1:
        log10_L_bol = np.array([np.log10((cte_eps_r/(1.-cte_eps_r))*9*10**20*accbh[p]/(3.16*10**7)*1.98892*10**33) for p in range(len(edd))])
    
    ## not so easy method
    if radia_efficient_agn==2:
        log10_L_bol=[]
        Ledd = np.array([1.26*10**38*p for p in mbh]) ## in erg/s  
        for pp in range(len(mbh)):
            if accbh[pp]/edd[pp]>0.1:
                log10_L_bol.append(np.log10((cte_eps_r/(1.-cte_eps_r))*9*10**20*accbh[pp]/(3.16*10**7)*1.98892*10**33))
            elif accbh[pp]/edd[pp]<=0.1:
                log10_L_bol.append(np.log10((10*accbh[pp]/edd[pp])**2*0.1*Ledd[pp]))
            else:     
                log10_L_bol.append(0.)
    
    log10_L_bol = np.array(log10_L_bol)
    
    ## bolometric correction from Hopkins+06 to go from bolometric luminosity to xray.
    corr_BC = np.array([10.83*(10**(log10_L_bol[p]-33.6)/1e10)**0.28+6.08*(10**(log10_L_bol[p]-33.6)/1e10)**-0.020 for p in range(len(edd))])
    
    log10_L_xray = np.array([(log10_L_bol[p]-33.6) - np.log10(corr_BC[p])+33.6 for p in range(len(edd))])
    
    return log10_L_bol,log10_L_xray
    
def compute_luminosity(mbh, accbh, method):
    '''This function calculates the BH luminosity through one of two methods. Required
       inputs are black hole mass (need for eddington fraction calculation) and instantaneous
       accretion rate onto the black hole. The third input dictates the method.
       
       radia_efficient_agn=1 for the easy method to compute luminosity 
       radia_efficient_agn=2 for a more sophisticated model including radiatively inefficient agn
       cte_eps_r=0.1 or (0.2) is the radiative efficiency
    
       This version is adapted to be more flexible (i.e. accept floats or np.arrays.)
    
       For more information see: Habouzit+19: Linking galaxy structural properties... 
    '''
    # initial check on shape of mbh and accbh arrays.
    if isinstance(mbh, np.ndarray):
        # since this is queried for multiple objects, using default function.  
        return compute_luminosity_MH(mbh, accbh, method)   
        
    elif isinstance(mbh, float):
        # calculating only for float. note this fails if int.
        mbh= mbh*1e10/h ## Msun                                                                                                                                    
        accbh= accbh*10.22 ## Msun/yr                                                                                                                              
        edd = 4 * np.pi* cte_G * cte_m_p* mbh /cte_eps_r/ cte_sigma_t/ cte_c*(3.154*10**7)
        fedd = np.log10(accbh/edd)
    
        ## easy method
        if method==1:
            log10_L_bol = np.log10((cte_eps_r/(1.-cte_eps_r))*9*10**20*accbh/(3.16*10**7)*1.98892*10**33)
    
        ## not so easy method
        if method==2:
            log10_L_bol=[]
            Ledd = 1.26*10**38*mbh ## in erg/s  
            if accbh[pp]/edd[pp]>0.1:
                log10_L_bol.append(np.log10((cte_eps_r/(1.-cte_eps_r))*9*10**20*accbh/(3.16*10**7)*1.98892*10**33))
            elif accbh[pp]/edd[pp]<=0.1:
                log10_L_bol.append(np.log10((10*accbh/edd)**2*0.1*Ledd))
            else:     
                log10_L_bol.append(0.)
    
        ## bolometric correction from Hopkins+06 to go from bolometric luminosity to xray.
        corr_BC = 10.83*(10**(log10_L_bol-33.6)/1e10)**0.28+6.08*(10**(log10_L_bol-33.6)/1e10)**-0.020 
        log10_L_xray = (log10_L_bol-33.6) - np.log10(corr_BC)+33.6 
        
        return log10_L_bol,log10_L_xray
    
    else:
        print('Input type not understood. Please supply np.arrays or floats.')
        return -np.inf, -np.inf
