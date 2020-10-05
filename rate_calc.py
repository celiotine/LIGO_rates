import pandas as pd
import numpy as np
import glob
import sys
import re
from astropy.cosmology import Planck15 as cosmo
import rate_functions as functions   ### Mike's code
import astropy.units as u
#----------------------------------------------------------------------------------                   

## constants for SFR + metallicity sampling
Zsun = 0.0142    ### Solar metallicity                                                                                           
lowZ = Zsun/200  ### lower bound on metallicity                                                             
highZ = 2*Zsun   ### upper bound on metallicity                                                            
sigmaZ = 0.5     ### sigma of the lognormal distribution about the mean metallicity

N_zbins = 100    ### number of redshift bins
zmin = 0         ### lower bound of redshift space
zmax = 20        ### upper bound of redshift space

N_mbins = 100   ### number of mass bins

dtp = 0.01       ## timestep of COSMIC runs
    

#----------------------------------------------------------------------------------
def calc_detection_prob(m1, m2, z_merge):
    
    ## constants that reflect LIGO design sensitivity
    d_L8 = 1  ## in Gpc
    M_8 = 10  ## in Msun
    SNR_thresh = 8

    ## approximate typical SNR from Fishbach et al. 2018
    M_chirp = (m1*m2)**(3/5)/(m1+m2)**(1/5)
    d_L = (1+z_merge)*cosmo.comoving_distance(z_merge).value
    rho_0 = 8*(M_chirp*(1+z_merge)/M_8)**(5/6)*d_L8/d_L

    ## analytic approximation for P(omega) from Dominik et al. 2015
    omega = SNR_thresh/rho_0
    P_omega = 0.374222*(1-omega)**2 + 2.04216*(1-omega)**4 - 2.63948*(1-omega)**8 + 1.223098*(1-omega)**10
    
    return P_omega

#---------------------------------------------------------------------------------- 

### path to folders for COSMIC metallicity runs          
COSMIC_path = "./*"
all_COSMIC_runs = glob.glob(COSMIC_path)

### array of metallicities, should match those of COSMIC_runs                                        
metallicities = [Zsun, Zsun/10, Zsun/5, Zsun/2]

### create log-spaced redshift bins
if zmin==0:
    # account for log-spaced bins
    zmin = 1e-3
zfbins = np.logspace(np.log10(zmin), np.log10(zmax), N_zbins+1)
zmbins = np.logspace(np.log10(zmin), np.log10(zmax), N_zbins+1)

all_population_rates = []


for Z, COSMIC_run in zip(metallicities, all_COSMIC_runs):

    ### get total sampled mass of this population
    Mtot = pd.read_hdf("dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_" + str(Z) + ".h5", key='mass_stars')

    ### bcm array for this population
    bcm =  pd.read_hdf("dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_" + str(Z) + ".h5", key='bcm')
    BBH_mergers = bcm.loc[bcm['merger_type']== "1414"]


    ### calculate N_BBHm(zf,zm,Z,Z_all) by iterating through low, hi zbins
    # work down from highest zbin
    # zbins[::-1] reverses log space so it is hi to low
    # zbin_low, zbin_high are bounds for a given bin
    # for each formation bin, find all systems that merge in each merger bin
    
    all_z_rates = []
    all_mbin_rates = []

    ## set-up bounds for mass bins
    m1bin_low = 0
    m2bin_low = 0
    m1bin_high = np.max(BBH_mergers['mass0_1']) 
    m2bin_high = np.max(BBH_mergers['mass0_2'])

    m1bins = np.linspace(m1bin_low, m1bin_high, N_mbins)
    m2bins = np.linspace(m2bin_low, m2bin_high ,N_mbins)
    
    for m1bin_low, m1bin_high in zip(m1bins[:-1], m1bins[1:]):
        for m2bin_low, m2bin_high in zip(m2bins[:-1], m2bins[1:]):
            
            BBHm_mbin = BBH_mergers.loc[(BBH_mergers['mass0_2']<=m2bin_high) & (BBH_mergers['mass0_2']>m2bin_low) & (BBH_mergers['mass0_1']<=m1bin_high) & (BBH_mergers['mass0_1']>m1bin_low)]

            for zfbin_low, zfbin_high in zip(zfbins[::-1][1:], zfbins[::-1][:-1]):
                for zmbin_low, zmbin_high in zip(zmbins[::-1][1:zfbin_low], zmbins[::-1][:zfbin_high]):

                    # assuming all systems are born in this formation bin, find how many merge in this merger bin
                    if zmbin_low==0:
                        # special treatment for the first merger bin
                        tdelay_min = 0
                        tdelay_max = cosmo.lookback_time(zmbin_high).to(u.Myr).value

                    else:
                        tdelay_min = cosmo.lookback_time(zmbin_low).to(u.Myr).value
                        tdelay_max = cosmo.lookback_time(zmbin_high).to(u.Myr).value
        
                    N_BBHm_CP = len(BBHm_mbin.loc[(BBHm_mbin['tphys']<=tdelay_max) & (BBHm_mbin['tphys']>tdelay_min)])

                    ### calculate redshift/metallicity weighting
                    # get redshift in the middle of this log-spaced formation redshift interval 
                    zf_mid = 10**(np.log10(zfbin_low) + (np.log10(zfbin_high)-np.log10(zfbin_low))/2.0)
                    zm_mid = 10**(np.log10(zmbin_low) + (np.log10(zmbin_high)-np.log10(zmbin_low))/2.0)

                    # get masses in the middle of each mass bin for detection prob calc
                    m1_mid = (m1bin_low + m1bin_high)/2.0
                    m2_mid = (m2bin_low + m2bin_high)/2.0

                    # the relative weight of this metallicity at this particular redshift
                    this_Z_prob = functions.metal_disp_z(zf_mid, Z)
                    ### sum of conditional probabilities: P(Z|zf_mid) 
                    all_Z_prob_sum = np.sum(functions.metal_disp_z(np.ones(len(metallicities))*zf_mid, metallicities))
                    N_BBHm = N_BBHm_CP*this_Z_prob/all_Z_prob_sum
                    f_zm = N_BBHm/Mtot

                    # cosmological factor
                    E_zm = (cosmo._Onu0*(1+zm_mid)**4 + cosmo._Om0*(1+zm_mid)**3 + cosmo._Ok0*(1+zm_mid)**2 + cosmo._Ode0)**(1./2)

                    ### calculate source-frame merger rate for this formation, merger redshift
                    ## detection rate for source with component masses m1, m2
                    p_det = calc_detection_prob(m1_mid, m2_mid, zm_mid)
                    this_Rate = f_zm * functions.sfr_z(zf_mid, mdl='2017') * p_det / ((1+zm_mid)*E_zm)
                    all_z_rates.append(this_Rate)  ## this is the merger rate for this mass bin and this redshift bin

        ## sum rates across all redshifts
        mbin_rate = np.sum(all_z_rates)
        all_mbin_rates.append(mbin_rate)

        ## empty the array for redshift-dependent rates
        all_z_rates = []
        

    ## to get final rate, sum across all mass bins for this population
    population_rate = np.sum(all_mbin_rates)
    all_population_rates.append(population_rate)

    all_mbin_rates = []
