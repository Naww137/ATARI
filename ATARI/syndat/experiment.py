

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:19:52 2022

@author: noahwalton
"""

import numpy as np     
import pandas as pd
import syndat
from copy import deepcopy
    


class experiment:
    
    def __init__(self,
                    energy_domain=None,
                    experiment_parameters = {} , 
                    input_options = {}, 
                                                     ):
        """
        Instantiates the experiment object

        To synthesize an experiment, you must provide open count data and a theoretical cross section. 
        Options and alternative experimental parameters are optional inputs.

        Parameters
        ----------
        open_data : DataFrame or str, optional
            Open count data. If passing a DataFrame, needs columns 'tof', 'bw', 'c', 'dc'. If passing a string, must be filepath to csv with format from Brown, et al.
            If empty (Default), the open count spectra will be approximated with an exponential function as detailed in Walton, et al.
        energy_domain : array-like, optional
            Energy domain of interest, can be just min/max or entire energy grid. If giving min/max, must be within the domain of the theoretical data.
            If giving energy grid, it must align with the energy grid given with the theoretical data.
            If empty (Default) the entire energy grid from the theoretical data will be used.
        experiment_parameters : dict, optional
            Experimental parameters alternative to default. Default parameters are described in Walton, et al., based on work by Brown, et al.,
            any parameters given here will replace the default parameters before the experiment is synthesized, by default {}
        options : dict, optional
            Keyword options, mostly for debugging, by default { 'Perform Experiment':True, 'Add Noise': True}
        """
        
        ### Default options
        default_options = { 'Add Noise':True, 
                            'Sample TURP':True,
                            'Sample TOCS':True, 
                            'Calculate Covariance':True,
                            'Compression Points':[],
                            'Grouping Factors':None, #[100]

                            'Smooth Open Spectrum':False } 

        ### redefine options dictionary if any input options are given
        options = default_options
        for old_parameter in default_options:
            if old_parameter in input_options:
                options.update({old_parameter:input_options[old_parameter]})
        for input_parameter in input_options:
            if input_parameter not in default_options:
                raise ValueError('User provided an unrecognized input option')
        self.options = options
        
        ### Gather options
        self.add_noise = options['Add Noise']
        self.sample_turp_bool = options['Sample TURP']
        self.sample_odat = options['Sample TOCS']
        self.smooth_open = options['Smooth Open Spectrum']
        self.calc_cov = options['Calculate Covariance']
        self.cpts = options['Compression Points']
        self.gfactors = options['Grouping Factors']

        ### Default experiment parameter dictionary
        default_exp = {
                        'n'         :   {'val'  :   0.067166,            'unc'  :   0},
                        'trigo'     :   {'val'  :   9758727,             'unc'  :   0},
                        'trigs'     :   {'val'  :   18476117,            'unc'  :   0},
                        'FP'        :   {'val'  :   35.185,              'unc'  :   0},
                        't0'        :   {'val'  :   3.326,               'unc'  :   0},
                        'bw'        :   {'val'  :   0.0064,              'unc'  :   0},
                        'm1'        :   {'val'  :   1,                   'unc'  :   0.016},
                        'm2'        :   {'val'  :   1,                   'unc'  :   0.008},
                        'm3'        :   {'val'  :   1,                   'unc'  :   0.018},
                        'm4'        :   {'val'  :   1,                   'unc'  :   0.005},
                        'a'         :   {'val'  :   582.7768594580712,   'unc'  :   np.sqrt(1.14395753e+03)},
                        'b'         :   {'val'  :   0.05149689096209191, 'unc'  :   np.sqrt(2.19135003e-05)},
                        'ab_cov'    :   {'val'  :   1.42659922e-1,       'unc'  :   None},
                        'ks'        :   {'val'  :   0.563,               'unc'  :   0.02402339737495515},
                        'ko'        :   {'val'  :   1.471,               'unc'  :   0.05576763648617445},
                        'b0s'       :   {'val'  :   9.9,                 'unc'  :   0.1},
                        'b0o'       :   {'val'  :   13.4,                'unc'  :   0.7}    }

        ### redefine experiment parameter dictionary if any new values are given
        pardict = default_exp
        for old_parameter in default_exp:
            if old_parameter in experiment_parameters:
                pardict.update({old_parameter:experiment_parameters[old_parameter]})
        self.pardict = pardict
        ### set reduction parameter attributes from input
        self.redpar = pd.DataFrame.from_dict(pardict, orient='index')


        ### define self.energy_grid upon init
        self.def_self_energy_grid(energy_domain)



    def run(self, theoretical_data,
                        open_data=None):
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        theoretical_data : DataFrame or str
            Theoretical cross section expected to be seen in the laboratory setting. If DataFrame needs columns 'E' and 'theo_trans'. If string, must be filepath to sammy.lst.

        Raises
        ------
        ValueError
            _description_
        """

        ### sample true underlying resonance parameters from measured values - defines self.theo_redpar
        self.sample_turp(self.pardict, self.sample_turp_bool)

        ### read in theoretical cross section/transmission data - defines self.sdat
        self.read_theoretical(theoretical_data)


        ### Check that energy grid aligns with theoretical and open
        if self.energy_domain is None:
            # if energy_domain given is None, use the one from the theoretical data
            self.energy_domain = self.sdat.E
        else:
            if len(self.energy_domain) != len(self.sdat.E):
                # if given energy domain is not equal to the theoretical, try to truncate
                self.sdat = self.sdat[(self.sdat.E>=min(self.energy_domain))&(self.sdat.E<=max(self.energy_domain))].reset_index(drop=True)
            if len(self.energy_domain) == 2:
                # if given energy domain is of len=2, take as min/max
                self.energy_domain = self.sdat.E
            if np.allclose(np.array(self.energy_domain), np.array(self.sdat.E), rtol=1e-08, atol=0, equal_nan=False):
                # check to make sure everything lines up 
                pass
            else:
                raise ValueError("An energy grid was given but it does not line up with that of the theoretical data")


        ### Decide on an open spectra
        if open_data is None:
            self.odat = self.approximate_open_spectra(self.energy_domain, self.smooth_open)
        else:
            if self.smooth_open == True:
                print("Warning: The user option 'Smooth Open Spectrum' was set to true but you are reading an open spectrum - this option will have no effect")
            self.read_odat(open_data)
            
        # sample a realization of the theoretical, true-underlying open count spectra
        self.sample_true_open_spectrum(self.sample_odat) 
        # calculate a vectorized true underlying background function
        self.theo_Bi = self.bkg(self.odat.tof,self.theo_redpar.val.a,self.theo_redpar.val.b)
        # generate raw count data for sample in given theoretical transmission and assumed true reduction parameters/open count data
        self.generate_sdat(self.add_noise)
        # reduce the experimental data
        self.reduce()
            


    def __repr__(self):
        return f"This experiment has the following options:\n{self.options}"


    # ----------------------------------------------------------
    #    Begin Methods
    # ----------------------------------------------------------

    def def_self_energy_grid(self, energy_domain):

        if energy_domain is None:
            # Keep as None, self.run() will take the energy grid from the given theoretical data
            self.energy_domain = energy_domain
        else:
            if len(energy_domain) == 2:
                # take an energy domain of len=2 as min/max and generate an energy grid linear in tof
                tof_min_max = syndat.exp_effects.e_to_t(np.array(energy_domain),self.redpar.val.FP, True)*1e6+self.redpar.val.t0 #micro s
                tof_grid = np.arange(min(tof_min_max), max(tof_min_max), self.redpar.val.bw )#micro s
                energy_domain = syndat.exp_effects.t_to_e((tof_grid-self.redpar.val.t0)*1e-6,self.redpar.val.FP,True) #back to s for conversion to eV
            if len(energy_domain) > 2:
                # if a vector is given, take that as the energy grid
                self.energy_domain = energy_domain
            elif len(energy_domain) == 1:
                raise ValueError("Energy domain is too small for tof bin width.")
            else:
                raise ValueError("Input for energy_domain not recognized")
            
        return

# --------------------------------------------------------------------------------------------------------------------------


    def read_theoretical(self, theoretical_data):
        """
        Reads in a theoretical cross section.

        Parameters
        ----------
        theoretical_data : DataFrame or str
            If DataFrame, must contain clumns 'E' and 'theo_trans'. If str, must be the full path to a sammy.lst file.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        
        # check types
        if isinstance(theoretical_data, pd.DataFrame):
            theo_df = theoretical_data
            if 'E' not in theo_df.columns:
                raise ValueError("Column name 'E' not in theoretical DataFrame passed to experiment class.")
            if 'theo_trans' not in theo_df.columns:
                raise ValueError("Column name 'theo_trans' not in theoretical DataFrame passed to experiment class.")
        elif isinstance(theoretical_data, str):
            theo_df = syndat.sammy_interface.readlst(theoretical_data)
        else:
            raise ValueError("Theoretical data passed to experiment class is neither a DataFrame or path name (string).")
        
        sdat = pd.DataFrame()
        sdat['theo_trans'] = theo_df.theo_trans #
        sdat['E'] = theo_df.E
        sdat['tof'] = syndat.exp_effects.e_to_t(sdat.E, self.redpar.val.FP, True)*1e6+self.redpar.val.t0
        sdat.sort_values('tof', axis=0, ascending=True, inplace=True)
        sdat.reset_index(drop=True, inplace=True)

        self.sdat = sdat
        self.theo = theo_df


# --------------------------------------------------------------------------------------------------------------------------
    
  
    def read_odat(self,open_data):
        """
        Reads in an open count dataset.

        Parameters
        ----------
        open_data : DataFrame or str
            If DataFrame, must contain clumns 'tof','bw', 'c', and 'dc'. If str, must be the full path to a csv file.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        
        # check types for what to do 
        if isinstance(open_data, pd.DataFrame):
            if 'tof' not in open_data.columns:
                raise ValueError("Column name 'tof' not in open count DataFrame passed to experiment class.")
            if 'bw' not in open_data.columns:
                raise ValueError("Column name 'bw' not in open count DataFrame passed to experiment class.")
            if 'c' not in open_data.columns:
                raise ValueError("Column name 'c' not in open count DataFrame passed to experiment class.")
            if 'dc' not in open_data.columns:
                raise ValueError("Column name 'dc' not in open count DataFrame passed to experiment class.")

            # calculate energy from tof and experiment parameters
            open_data['E'] = syndat.exp_effects.t_to_e((open_data.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True) 
            odat = open_data
            
        elif isinstance(open_data, str):
            # -------------------------------------------
            # the below code takes open data from Jesse's csv can gets it into the correct format, rather, I want to make thid function take the proper format
            # -------------------------------------------
            odat = pd.read_csv(open_data, sep=',') 
            odat = odat[odat.tof >= self.redpar.val.t0]
            odat.sort_values('tof', axis=0, ascending=True, inplace=True)
            odat.reset_index(drop=True, inplace=True)
            odat['E'] = syndat.exp_effects.t_to_e((odat.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True) 
            odat['bw'] = odat.bin_width*1e-6 
            odat.rename(columns={"counts": "c", "dcounts": "dc"}, inplace=True)
            # -------------------------------------------
        else:
            raise TypeError("Open data passed to experiment class is not of type DataFrame or string (pathname).")


        # filter to energy limits
        if len(odat.E) != len(self.energy_domain):
            # must round to match .LST precision                    # round to the same digits as max(self.energy_domain)
            odat = odat[(round(odat.E,6)>=min(round(self.energy_domain, 6)))&(round(odat.E,6)<=max(round(self.energy_domain, 6)))].reset_index(drop=True)
        if np.allclose(np.array(odat.E), np.array(self.energy_domain)):
            pass
        else:
            raise ValueError("The open data's energy grid does not align with the defined experiment.energy_domain")

        # Define class attribute
        self.odat = odat
        

# --------------------------------------------------------------------------------------------------------------------------

        
    def bkg(self,tof,a,b):
        return a*np.exp(tof*-b)


# --------------------------------------------------------------------------------------------------------------------------

    def sample_true_open_spectrum(self, bool):
        
        theo_odat = self.odat.copy()
        # realization_of_true_cts = syndat.exp_effects.pois_noise(theo_odat.c)

        if bool:
            cycles = 35
            theo_cycle_data = theo_odat.c/cycles
            monitor_factors = np.random.default_rng().normal(1,0.0160*2, size=cycles)
            noisy_cycle_data = syndat.exp_effects.pois_noise(theo_cycle_data)*monitor_factors[0]
            for i in range(cycles-1):
                noisy_cycle_data += syndat.exp_effects.pois_noise(theo_cycle_data)*monitor_factors[i]
            # experiment.sdat.c = noisy_cycle_data

            theo_odat['c'] = noisy_cycle_data
            theo_odat['dc'] = np.sqrt(noisy_cycle_data)

        self.theo_odat = theo_odat

    
    def sample_turp(self, redpar, bool):
        
        theo_redpar = deepcopy(redpar)

        if bool:
            for par in theo_redpar:
                if par == 'ab_cov':
                    continue
                # TODO: determine if I should be sampling monitor corrections
                # if par == 'm1':
                #     continue
                # if par == 'm2':
                #     continue
                # if par == 'm3':
                #     continue
                # if par == 'm4':
                #     continue
                theo_redpar[par]['val'] = np.random.default_rng().normal(theo_redpar[par]['val'], theo_redpar[par]['unc'])
        
        self.theo_redpar = pd.DataFrame.from_dict(theo_redpar, orient='index')

# --------------------------------------------------------------------------------------------------------------------------


    def generate_sdat(self, add_noise):
        """
        Generates a set of noisy, sample in count data from a theoretical cross section via the novel un-reduction method (Walton, et al.).

        Parameters
        ----------
        add_noise : bool
            Whether or not to add noise to the generated sample in data.

        Raises
        ------
        ValueError
            _description_
        """

        if len(self.theo_odat) != len(self.sdat):
            raise ValueError("Experiment open data and sample data are not of the same length, check energy domain")

        monitor_array = [self.theo_redpar.val.m1, self.theo_redpar.val.m2, self.theo_redpar.val.m3, self.theo_redpar.val.m4]

        self.sdat, self.theo_c = syndat.exp_effects.inverse_reduction(self.sdat, self.theo_odat, add_noise, self.sample_turp_bool,
                                                                self.theo_redpar.val.trigo, self.theo_redpar.val.trigs, 
                                                                self.theo_redpar.val.ks,self.theo_redpar.val.ko, 
                                                                self.theo_Bi, self.theo_redpar.val.b0s, self.theo_redpar.val.b0o, 
                                                                monitor_array)
        

# --------------------------------------------------------------------------------------------------------------------------

    def reduce(self):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        """

        if self.gfactors is not None:
            # Re-bin the data according to new structure
            grouped_odat = syndat.exp_effects.regroup(self.odat.tof, self.odat.c, self.gfactors, self.cpts)
            grouped_sdat = syndat.exp_effects.regroup(self.sdat.tof, self.sdat.c, self.gfactors, self.cpts)
            odat = pd.DataFrame(grouped_odat, columns=['tof','bw','c','dc'])
            sdat = pd.DataFrame(grouped_sdat, columns=['tof','bw','c','dc'])

            # calculate energy and redefine experiment.odat/sdat with the regrouped data
            odat['E'] = syndat.exp_effects.t_to_e((odat.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True)
            sdat['E'] = syndat.exp_effects.t_to_e((odat.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True) 
            self.odat = odat
            self.sdat = sdat

        # create transmission object
        self.trans = pd.DataFrame()
        self.trans['tof'] = self.sdat.tof
        self.trans['E'] = self.sdat.E
        # self.trans['theo_trans'] = self.sdat.theo_trans

        # get count rates for sample in data
        # self.sdat['cps'], self.sdat['dcps'] = syndat.exp_effects.cts_to_ctr(self.sdat.c, self.sdat.dc, self.sdat.bw*1e-6, self.redpar.val.trigs)
        # self.odat['cps'], self.odat['dcps'] = syndat.exp_effects.cts_to_ctr(self.odat.c, self.odat.dc, self.odat.bw*1e-6, self.redpar.val.trigs)

        # estimated background function
        self.Bi = self.bkg(self.odat.tof*1e6,self.redpar.val.a,self.redpar.val.b) # calc bkg again to recalculate Bi on restructured grid

        # define systematic uncertainties
        sys_unc = self.redpar.unc[['a','b','ks','ko','b0s','b0o','m1','m2','m3','m4']].astype(float)
        monitor_array = [self.redpar.val.m1, self.redpar.val.m2, self.redpar.val.m3, self.redpar.val.m4]

        self.trans['exp_trans'], unc_data, rates = syndat.exp_effects.reduce_raw_count_data(self.sdat.tof, 
                                                                                    self.sdat.c, self.odat.c, self.sdat.dc, self.odat.dc,
                                                                                    self.odat.bw, self.redpar.val.trigo, self.redpar.val.trigs, self.redpar.val.a,self.redpar.val.b, 
                                                                                    self.redpar.val.ks, self.redpar.val.ko, self.Bi, self.redpar.val.b0s,
                                                                                    self.redpar.val.b0o, monitor_array, sys_unc, self.redpar.val.ab_cov, self.calc_cov)
        
        self.CovT, self.CovT_stat, self.CovT_sys = unc_data
        if self.calc_cov:
            self.trans['exp_trans_unc'] = np.sqrt(np.diag(self.CovT))
        else:
            self.trans['exp_trans_unc'] = np.sqrt(self.CovT)
            

        # define data cps
        self.odat['cps'] = rates[0]
        self.odat['dcps'] = rates[1]
        self.sdat['cps'] = rates[2]
        self.sdat['dcps'] = rates[3]

# --------------------------------------------------------------------------------------------------------------------------


    def approximate_open_spectra(self, energy_grid, smooth_open):

        def open_count_rate(tof):
            return (2212.70180199 * np.exp(-3365.55134779 * tof*1e-6) + 23.88486286) 

        tof = syndat.exp_effects.e_to_t(energy_grid,self.redpar.val.FP,True)*1e6+self.redpar.val.t0 # microseconds

        # calculate a tof count rate spectra, convert to counts, add noise 
        cps_open_approx = open_count_rate(tof)
        bin_width = abs(np.append(np.diff(tof), np.diff(tof)[-1])*1e-6)
        cts_open_approx = cps_open_approx*bin_width*self.redpar.val.trigo
        if smooth_open:
            cts_open_measured = cts_open_approx
        else:
            cts_open_measured = syndat.exp_effects.pois_noise(cts_open_approx)

        open_dataframe = pd.DataFrame({'tof'    :   tof,
                                        'bw'    :   bin_width,
                                        'c'     :   cts_open_measured,
                                        'dc'    :   np.sqrt(cts_open_measured)})

        open_dataframe['E'] = syndat.exp_effects.t_to_e((open_dataframe.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True) 

        return open_dataframe


    

