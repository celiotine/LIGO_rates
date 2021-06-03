import pandas as pd
import numpy as np
import glob
import sys
import re
from astropy.cosmology import Planck15 as cosmo
import rate_functions as functions   ### Mike's code                                           
import astropy.units as u
import scipy

#-----------------------------------------------------------------------------------#

## analytic approximation for P(omega) from Dominik et al. 2015   
def P_omega(omega_values):
    return 0.374222*(1-omega_values)**2 + 2.04216*(1-omega_values)**4 - 2.63948*(1-omega_values)**8 + 1.222543*(1-omega_values)**10

#-----------------------------------------------------------------------------------# 

### Monte Carlo sampling for detections above the given SNR threshold
def calc_detection_prob(m1, m2, z_merge):
    
    ## constants that reflect LIGO design sensitivity                                      
    d_L8 = 1  ## in Gpc                                                             
    M_8 = 10  ## in Msun                                                                                                      
    SNR_thresh = 8

    ## approximate typical SNR from Fishbach et al. 2018                                         
    M_chirp = (m1*m2)**(3./5)/(m1+m2)**(1./5)
    d_C = cosmo.comoving_distance(z_merge).to(u.Gpc).value
    d_L = (1+z_merge)*d_C
    
    rho_0 = 8*(M_chirp*(1+z_merge)/M_8)**(5./6)*d_L8/d_L   ## this is the "typical/optimal" SNR
    if (rho_0 < SNR_thresh): return 0
    
    
    ## sample omega according to distribution for omega
    ## sample omega according to distribution for omega via inverse CDF method
    dist_size = 10000
    sample_size = 1000
     
    P_omega_dist = P_omega(np.linspace(0, 1, dist_size))
    inv_P_omega = scipy.interpolate.interp1d(P_omega_dist, np.linspace(0, 1, dist_size), fill_value="extrapolate")
    omega = inv_P_omega(np.random.uniform(0, 1, sample_size))

    ## find the true SNRs given sky location
    rho = omega*rho_0
    accept_SNR_num = len(rho[np.where(rho >= SNR_thresh)])
    
    p_det = accept_SNR_num/sample_size
              
    return p_det

#-----------------------------------------------------------------------------------# 
## calculate the mass fraction of mergers for this merger redshift bin
def find_f_z(pop, zmbin_low, zmbin_high, zf_mid, this_Z, mets, M_tot):
    
    Zsun = 0.017
    sigmaZ = 0.5
    lowZ = Zsun/200
    highZ = 2*Zsun
    
    # assuming all systems are born in this formation bin, find how many merge in this merger bin              
    if zmbin_low==0:
        # special treatment for the first merger bin                                                                
        tdelay_max = 0
        tdelay_min = cosmo.lookback_time(zmbin_high).to(u.Myr).value

    else:
        tdelay_max = cosmo.lookback_time(zf_mid).to(u.Myr).value - cosmo.lookback_time(zmbin_low).to(u.Myr).value
        tdelay_min = cosmo.lookback_time(zf_mid).to(u.Myr).value - cosmo.lookback_time(zmbin_high).to(u.Myr).value
        

    N_merge_CP = len(pop.loc[(pop['tphys']<=tdelay_max) & (pop['tphys']>tdelay_min)])
        
        
    # the relative weight of this metallicity at this particular redshift                                    
    this_Z_prob = functions.metal_disp_z(zf_mid, this_Z, sigmaZ, lowZ, highZ)
                    
    ### sum of conditional probabilities: P(Z|zf_mid)                                                                            
    all_Z_prob_sum = np.sum(functions.metal_disp_z(np.ones(len(mets))*zf_mid, mets, sigmaZ, lowZ, highZ))
    N_merge = N_merge_CP*this_Z_prob/all_Z_prob_sum

    f_zm = float(N_merge)/M_tot
        
    return f_zm

#-----------------------------------------------------------------------------------# 

## calculate the detection rate for the given population
## returns array of detection rates across mass bins
def detection_rates(population, this_Z, M_tot, mets, m1bins, m2bins, zfbins, zmbins):
    
    f_loc_Z = []

    ## constants
    c = 3e8   ## speed of light in m/s
    prefactor = 4*np.pi*c / cosmo.H(0).to(u.m * u.Gpc**-1 * u.s**-1).value
    
    merge_z_rates = []   ## merger rates for each redshift bin
    merge_mbin_rates = []  ## merger rates, summed over redshifts, for each mass bin
    
    for m1bin_low, m1bin_high in zip(m1bins[:-1], m1bins[1:]):
        for m2bin_low, m2bin_high in zip(m2bins[:-1], m2bins[1:]):
            

            merge_mbin = population.loc[(population['mass0_2']<=m2bin_high) & (population['mass0_2']>m2bin_low) \
                                        & (population['mass0_1']<=m1bin_high) & (population['mass0_1']>m1bin_low)]

            #print ("# mergers in this mass bin: ", len(merge_mbin))
            for zfbin_low, zfbin_high in zip(zfbins[::-1][1:], zfbins[::-1][:-1]):

                zmbin_low_max = np.where(zmbins == zfbin_low)[0][0]
                zmbin_high_max = np.where(zmbins == zfbin_high)[0][0]
                
                for zmbin_low, zmbin_high in zip(zmbins[::-1][1:zmbin_low_max], zmbins[::-1][:zmbin_high_max]):
                            
                    ### calculate redshift/metallicity weighting                                                                                                                                   
                    # get redshift in the middle of this log-spaced formation redshift interval                                                                                                    
                    zf_mid = 10**(np.log10(zfbin_low) + (np.log10(zfbin_high)-np.log10(zfbin_low))/2.0)
                    zm_mid = 10**(np.log10(zmbin_low) + (np.log10(zmbin_high)-np.log10(zmbin_low))/2.0)
                    
                    # get masses in the middle of each mass bin for detection prob calc                                                                                                            
                    m1_mid = (m1bin_low + m1bin_high)/2.0
                    m2_mid = (m2bin_low + m2bin_high)/2.0
                    

                    f_zm = find_f_z(merge_mbin, zmbin_low, zmbin_high, zf_mid, this_Z, mets, M_tot)
                    
                    # cosmological factor                                                                                                               
                    E_zm = (cosmo._Onu0*(1+zm_mid)**4 + cosmo._Om0*(1+zm_mid)**3 + cosmo._Ok0*(1+zm_mid)**2 + \
                            cosmo._Ode0)**(1./2)

                    ### calculate source-frame merger rate for this formation, merger redshift                                                    
                    ## detection rate for source with component masses m1, m2                                                                       
                    p_det = calc_detection_prob(m1_mid, m2_mid, zm_mid)
                    
                    SFR = functions.sfr_z(zf_mid, mdl='2017')*(u.Mpc**-3).to(u.Gpc**-3)
                    D_c = cosmo.comoving_distance(zm_mid).to(u.Gpc).value
                    
                    ## merger rate for this [zf, zm] in yr^-1
                    this_merge_Rate =  prefactor * f_zm * SFR* p_det * D_c**2 / ((1+zm_mid)*E_zm)* \
                    (zmbin_high - zmbin_low)
                    
                    #print (this_merge_Rate)
                    merge_z_rates.append(this_merge_Rate)

            ## sum rates across all redshifts                                                                                                       
            merge_mbin_rate = np.sum(merge_z_rates)
            merge_mbin_rates.append(merge_mbin_rate)

            #print ("merger rate for m1=", m1_mid, " m2=", m2_mid, ": ", merge_mbin_rate
            ## empty the array for redshift-dependent rates                                                                                                        
       
            merge_z_rates = []
    
    return merge_mbin_rates, f_loc_Z    

#-----------------------------------------------------------------------------------# 

### path to folder with COSMIC metallicity runs                                              
COSMIC_path = "pessimistic_CE_runs/*"
all_COSMIC_runs = glob.glob(COSMIC_path)

## data frame to store rate output
columns=['metallicity', 'BBH_detection_rate', 'tot_detection_rate']
df = pd.DataFrame(columns=columns)

## define parameters
Zsun = 0.017    ## Solar metallicity                                          

N_zbins = 100    ## number of redshift bins                                                                  
zmin = 0         ## lower bound of redshift space                                
zmax = 20        ## upper bound of redshift space

N_mbins_BBH = 10    ## number of mass bins   
N_mbins_tot = 10     
m1bin_low_BBH = 0
m2bin_low_BBH = 0
m1bin_low_tot = 0
m2bin_low_tot = 0 
m1bin_high_BBH = 100
m2bin_high_BBH = 100
m1bin_high_tot = 100
m2bin_high_tot = 100

m1bins_BBH = np.linspace(m1bin_low_BBH, m1bin_high_BBH, N_mbins_BBH)
m2bins_BBH = np.linspace(m2bin_low_BBH, m2bin_high_BBH ,N_mbins_BBH)
m1bins_tot = np.linspace(m1bin_low_tot, m1bin_high_tot, N_mbins_tot)
m2bins_tot = np.linspace(m2bin_low_tot, m2bin_high_tot ,N_mbins_tot)

### array of metallicities, should match those of COSMIC_runs                         
metallicities = [Zsun/200, Zsun/100, Zsun/20, Zsun/10, Zsun/5, Zsun/2, Zsun]

### create log-spaced redshift bins                                    
if zmin==0:
    # account for log-spaced bins                            
    zmin = 1e-3
zfbins = np.logspace(np.log10(zmin), np.log10(zmax), N_zbins+1)
zmbins = np.logspace(np.log10(zmin), np.log10(zmax), N_zbins+1)


## loop through all COSMIC populations and calculate BBHm merger rates

## 2D arrays storing mass distribution and f_loc values for all metallicites
all_BBH_pop_mbin_rates = []
all_tot_pop_mbin_rates = []
all_BBH_f_loc = []
all_tot_f_loc = []

read_mets = []

for pop_dir in all_COSMIC_runs:
    
    data = glob.glob(pop_dir + "/*.h5")[0]
    this_Z_frac = float(re.findall('\d+.\d+', data)[0]) ## parse population metallicity from filename
    this_Z = round(this_Z_frac*Zsun, 6)
    read_mets.append(this_Z_frac)
                        
    ### get total sampled mass of this population                    
    Mtot_arr = pd.read_hdf(data, key='mass_stars') 
    Mtot = Mtot_arr.iloc[-1].values[0]
    
    bpp =  pd.read_hdf(data, key='bpp')

    ## find BBH mergers using bcm array
    bcm = pd.read_hdf(data, key='bcm')
    BBHm_bcm = bcm.loc[bcm["merger_type"] == "1414"] 
    BBHm_bin = BBHm_bcm['bin_num'].values
    BBHm_bpp = bpp.loc[BBHm_bin] 
    

    all_mergers = bpp.loc[(bpp['evol_type']==6)]
    BBH_mergers = BBHm_bpp.loc[BBHm_bpp["evol_type"]==6] 
    
    ## detection rates for this metallicity population, separated by mass bin
    mbin_BBH_rate_array, f_loc_BBH_array = detection_rates(BBH_mergers, this_Z, Mtot, metallicities, m1bins_BBH, m2bins_BBH, zfbins, zmbins)
    mbin_tot_rate_array, f_loc_tot_array = detection_rates(all_mergers, this_Z, Mtot, metallicities, m1bins_tot, m2bins_tot, zfbins, zmbins)
    
    print ("population: ", pop_dir)
    print ("total BBH detection rate: ", np.sum(mbin_BBH_rate_array))
    print ("total DCO detection rate: ", np.sum(mbin_tot_rate_array))
    
    all_BBH_pop_mbin_rates.append(mbin_BBH_rate_array)
    all_tot_pop_mbin_rates.append(mbin_tot_rate_array)
    all_BBH_f_loc.append(f_loc_BBH_array)
    all_tot_f_loc.append(f_loc_tot_array)

    mbin_BBH_rate_array = []
    mbin_tot_rate_array = []
    f_loc_BBH_array = []
    f_loc_tot_array = []

## to get final rates for each population, sum across all mass bins   
j = 0
for BBH_rate_vals, tot_rate_vals in zip(all_BBH_pop_mbin_rates, all_tot_pop_mbin_rates):  
    full_BBH_rate = np.sum(BBH_rate_vals)
    full_tot_rate = np.sum(tot_rate_vals)
    this_df = pd.DataFrame([[read_mets[j], full_BBH_rate, full_tot_rate]], columns=columns)
    df = df.append(this_df, sort=False, ignore_index=True)
    j += 1
df.to_csv("Z_pop_rates.csv", index=False)
