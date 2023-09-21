# %%
import numpy as np
import pandas as pd
import os
import h5py
import scipy.stats as sts
import copy

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.xs import SLBW
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.utils.stats import chi2_val
from ATARI.theory.resonance_statistics import make_res_par_avg
from ATARI.theory.experimental import xs_2_trans, trans_2_xs
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions

import sys
sys.path.insert(0, '/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/')

from scipy import integrate
import functions as fn 
import classes as cls

# ================================================================================================
# Functions for setting up the feature bank and constriants
# ================================================================================================

def get_parameter_grid(energy_grid, res_par_avg, num_Er, num_Gt, option=0):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_grid) + res_par_avg['Gt99']*1e-3
    min_Elam = min(energy_grid) - res_par_avg['Gt99']*1e-3

    if option < 2:
        num_Gn = 1
        Gn = np.array([1])
    else:
        num_Gn = num_Gt
        Gn = np.logspace(np.log10(res_par_avg['Gn01']), np.log10(res_par_avg['Gn99']), num_Gn)
                
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    Gt = np.logspace(np.log10(res_par_avg['Gt01']), np.log10(res_par_avg['Gt99']), num_Gt)    
    
    return Er, Gt, Gn

def get_resonance_ladder_from_parameter_grid(Er_list, Gt_list, Gn_list):
    Er_list = np.array(Er_list); num_Er = len(Er_list)
    Gt_list = np.array(Gt_list); num_Gt = len(Gt_list)
    Gn_list = np.array(Gn_list); num_Gn = len(Gn_list)

    Gn_matrix = np.tile(Gn_list, num_Gt*num_Er)
    Gt_matrix = np.tile(np.repeat(Gt_list, num_Gn), num_Er)
    Er_matrix = np.repeat(Er_list, num_Gt*num_Gn)

    resonance_ladder_array = np.column_stack((Er_matrix, Gt_matrix, Gn_matrix))
    return resonance_ladder_array


def get_feature(E, Res, particle_pair, kinematic_constant, phi, lwave, P):
    Elam=Res[0]; Gtot=Res[1]*1e-3; Gn=Res[2]*1e-3
    _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    PPElam = P/PElam
    # A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
    #                                         -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
    A_column =  kinematic_constant*Gn* ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                    - (Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
    return A_column
            
def get_feature_bank_from_resonance_ladder(E, particle_pair, resonance_ladder_array, solver, sammy_RTO=None):
    E = np.sort(np.array(E))
    number_of_resonances = len(resonance_ladder_array)
    Resonance_Matrix = np.zeros((len(E), number_of_resonances))

    lwave = 0
    _, P, phi, k = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    g = gstat(3.0, particle_pair.I, particle_pair.i)
    kinematic_constant = (4*np.pi*g/k**2)
    potential_scattering = kinematic_constant * np.sin(phi)**2 

    if solver == "syndat_SLBW":
        for iRes, Res in enumerate(resonance_ladder_array):
            Resonance_Matrix[:, iRes] = get_feature(E, Res, particle_pair, kinematic_constant, phi, lwave, P)

    elif solver == "sammy":
        if sammy_RTO is None:
            raise ValueError(f"Solver selected: {solver} but user did not pass sammy_RTO")
        else:
            for iRes, Res in enumerate(resonance_ladder_array):
                sammyINP = sammy_classes.SammyInputData(particle_pair=particle_pair, resonance_ladder=pd.DataFrame([np.append(Res,[3.0,1,0,1])], columns=["E","Gt","Gn","J","chs","lwave","J_ID"], index=[0]), 
                                                        energy_grid=E, temp = 304.5, FP=35.185, frac_res_FP=0.049600, target_thickness=0.067166)
                sammyOUT = sammy_functions.run_sammy(sammyINP, sammy_RTO)
                pw = sammyOUT.pw
                if sammy_RTO.reaction == "total" or sammy_RTO.reaction == "transmission":
                    Resonance_Matrix[:, iRes] = pw.theo_xs.values-potential_scattering.flatten()
                    if np.any(np.isinf(pw.theo_xs-potential_scattering.flatten())):
                        _=0
                else:
                    Resonance_Matrix[:, iRes] = pw.theo_xs.values

    else:
        raise ValueError(f"Solver: {solver} not recognized")

    return Resonance_Matrix, potential_scattering.flatten(), E


#%%

ac = 0.81271  # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider

Gg_DOF = 10

from ATARI.theory import scattering_params
shift, penetration, phi, k = scattering_params.FofE_explicit([1,2500], ac, M, m, 0)
res_par_avg = make_res_par_avg(D_avg = 8.79, 
                            Gn_avg= 2, #np.mean(penetration)*2*46.4, 
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = Gg_DOF, 
                            print = False)

spin_groups = [ (3.0,1,0) ]
average_parameters = {'3.0':res_par_avg}

Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )   

#%% Fitting functions

from scipy.optimize import minimize
from scipy.optimize import fsolve, least_squares

def get_lsw(approx, exact):
    data = exact
    data_unc = np.ones_like(data)*1e-5

    def f(x):
        # return np.sum(((approx*x-data)/data_unc)**2)
        return np.sum(((approx*x-data))**2)
    
    out = minimize(f, x0=1)
    return out.x, data, data_unc


def get_GnGt(i):
    iGn = (i % ipar)
    iGt = int((i-iGn)/ipar)

    Gn = Gn_list[iGn]
    Gt = Gt_list[iGt]
    return Gn, Gt


# %%

# reference energy ranges and bin widths
# E/Bw = 0-6eV / 0.8192 us
# E/Bw = 7-125eV / 0.1024 us
# E/Bw = 125-2500eV / 0.0256 us

# energies = [4,            10,50,100,              150,500,1000,1500,2000,2500]
# bw = [0.8192,   0.1024,0.1024,0.1024,    0.0256,0.0256,0.0256,0.0256,0.0256,0.0256]

energies = [50,100,              500,1000,1500,2000,2500]
bw = [0.1024,0.1024,    0.0256,0.0256,0.0256,0.0256,0.0256]

save_Gg_error_cap = []
save_Gg_error_tot = []
save_Gn_error_cap = []
save_Gn_error_tot = []
save_cap_residual_cap =[]
save_cap_residual_tot =[]
save_sim_tot = []
save_sim_cap = []


for e,b in zip(energies, bw):
    print(f"Running E/bw = {e}/{b}")
    # build energy grid
    E_min_max = [np.max([e-20, 1.0]), e+20]
    energy_grid = E_min_max
    input_options = {'Add Noise': True,
                    'Calculate Covariance': False,
                    'Compression Points':[],
                    'Grouping Factors':None}
    experiment_parameters = {'bw': {'val':b,   'unc'   :   0}}
    exp = Experiment(energy_grid, 
                            input_options=input_options, 
                            experiment_parameters=experiment_parameters)

    ### sample different width combinations
    E1 = np.sort(exp.energy_domain)
    ipar = 10
    Er_list, Gt_list, Gn_list = get_parameter_grid(E1, res_par_avg, 1, ipar, option=2)
    Elam = e
    Er_list = [Elam]
    iElam = np.searchsorted(E1, Elam)
    # maxxs, _ = trans_2_xs(0.01, 0.00566)
    resonance_ladder_array = get_resonance_ladder_from_parameter_grid(Er_list, Gt_list, Gn_list)
    
    ### Sammy setup
    sammyRTO_cap = sammy_classes.SammyRunTimeOptions(
        path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
        model = 'SLBW',
        reaction = 'capture',
        sammy_runDIR = '/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/SAMMY_runDIR',
        shell = 'zsh',
        inptemplate = 'allexpcap_1sg.inp',
        keep_runDIR = False
        )
    sammyRTO_tot = copy.copy(sammyRTO_cap)
    sammyRTO_tot.inptemplate = 'allexptot_1sg.inp'
    sammyRTO_tot.reaction = 'total'

    ### Build resonances
    resonance_bank_cap, pscat_cap, E_cap = get_feature_bank_from_resonance_ladder(E1, 
                                                                Ta_pair, 
                                                                resonance_ladder_array,
                                                                "sammy", sammy_RTO=sammyRTO_cap)
    resonance_bank_tot, pscat_tot, E_tot = get_feature_bank_from_resonance_ladder(E1, 
                                                                Ta_pair, 
                                                                resonance_ladder_array,
                                                                "sammy", sammy_RTO=sammyRTO_tot)

    ### Calculate similarity coefficient
    norm_cap = np.atleast_2d(np.linalg.norm(resonance_bank_cap, axis=0))
    sim_cap =  np.dot(resonance_bank_cap.T,resonance_bank_cap)/(norm_cap.T @ norm_cap)

    norm_tot = np.atleast_2d(np.linalg.norm(resonance_bank_tot, axis=0))
    sim_tot =  np.dot(resonance_bank_tot.T,resonance_bank_tot)/(norm_tot.T @ norm_tot)

    ### find least similar case for both tot and cap to have as true and estimated feature
    imin_cap1, imin_cap2 = np.unique(np.argwhere(sim_cap == np.min(sim_cap)).flatten())
    imin_tot1, imin_tot2 = np.unique(np.argwhere(sim_tot == np.min(sim_tot)).flatten())
    # iest = 1 #9*ipar -1
    # itrue = 99 #9*ipar+ipar -1

    # loop over possibilities for true and estimated features
    # for iest, itrue in zip([imin_cap1, imin_cap2, imin_tot1, imin_tot2], [imin_cap2, imin_cap1, imin_tot2, imin_tot1]):
    iestbase = 9
    for iest, itrue in zip(np.ones(25)*iestbase, np.random.randint(2,100, 25)):

        try:
            print(f"iest/itrue = {iest}/{itrue}")
            iest = int(iest)
            itrue = int(itrue)
            # # capture
            approx_cap = resonance_bank_cap[:,iest]
            exact_cap = resonance_bank_cap[:,itrue]
            # total
            approx_tot = resonance_bank_tot[:,iest]
            exact_tot = resonance_bank_tot[:,itrue]

            ### Get ls weight for both reactions cap and tot
            wc, dc, dc_unc = get_lsw(approx_cap, exact_cap)
            wt, dt, dt_unc = get_lsw(approx_tot, exact_tot)
            dt = dt + pscat_tot

            ### Redefine approximations
            approx_cap = approx_cap*wc
            approx_tot = approx_tot*wt + pscat_tot
            exact_tot = exact_tot + pscat_tot
            Gn_true, Gt_true = get_GnGt(itrue)
            Gn_est, Gt_est = get_GnGt(iest)
            x_est = np.array([Gt_est-Gn_est, Gn_est])*1e-3

            ### Fit in SLBW, no-exp space
            Efit = np.array(E_cap[[iElam-1, iElam, iElam+1]], dtype=float)
            lwave = 0
            _, P, phi, k = FofE_recursive(Efit, Ta_pair.ac, Ta_pair.M, Ta_pair.m, lwave)
            g = gstat(3.0, Ta_pair.I, Ta_pair.i)

            def fcap(x):
                return (np.pi*g/k**2) * x[0]*x[1]/( (Efit-Elam)**2 + ((x[0]+x[1])/2)**2 )
            def ftot(x):
                Gt = x[0]+x[1]
                d = (Efit-Elam)**2 + (x[0]+x[1])**2/4
                return (4*np.pi*g/k**2) * (Gt*x[1]/(4*d)*np.cos(2*phi) + (Efit-Elam)*x[1]/(2*d)*np.sin(2*phi) )
            def fcap_diff(x):
                return fcap(x) - fcap(x_est)*wc
            def ftot_diff(x):
                return ftot(x) - ftot(x_est)*wt
            
            def func(x):
                return np.append(fcap_diff(x),ftot_diff(x))
            x0 = np.array([1, 1])*1e-3
            ls_sol = least_squares(func, x0, bounds=(0, 10))

            ### Pull out fit in SLBW/no-exp space
            Gg_fit = ls_sol.x[0]*1e3
            Gn_fit = ls_sol.x[1]*1e3

            est_ladder = pd.DataFrame(np.atleast_2d(np.append(np.array([Elam, Gt_est-Gn_est, Gn_est]),[3.0,1,0,1])), columns=["E","Gg","Gn","J","chs","lwave","J_ID"], index=[0])
            par_ladder = pd.DataFrame(np.atleast_2d(np.append([Elam, Gg_fit, Gn_fit],[3.0,1,0,1])), columns=["E","Gg","Gn","J","chs","lwave","J_ID"], index=[0])
            true_ladder = pd.DataFrame(np.atleast_2d(np.append(np.array([Elam, Gt_true-Gn_true, Gn_true]),[3.0,1,0,1])), columns=["E","Gg","Gn","J","chs","lwave","J_ID"], index=[0])

            ### Calculate estimate, parameterized estimate, and true
            sammyINP = sammy_classes.SammyInputData(particle_pair       = Ta_pair, 
                                                    resonance_ladder    = est_ladder,
                                                    energy_grid         =  E_cap,
                                                    initial_parameter_uncertainty = 0.1,
                                                    temp = 304.5, FP=35.185, frac_res_FP=0.049600, target_thickness=0.067166)

            # pw_cap_est, par_cap_est = sammy_functions.run_sammy(sammyINP, sammyRTO_cap)
            # pw_tot_est, par_tot_est = sammy_functions.run_sammy(sammyINP, sammyRTO_tot)

            # sammyINP.resonance_ladder = par_ladder
            # pw_cap_par, par_cap_par = sammy_functions.run_sammy(sammyINP, sammyRTO_cap)
            # pw_tot_par, par_tot_par = sammy_functions.run_sammy(sammyINP, sammyRTO_tot)

            sammyINP.resonance_ladder = true_ladder
            sammyOUT_cap = sammy_functions.run_sammy(sammyINP, sammyRTO_cap)
            sammyOUT_tot = sammy_functions.run_sammy(sammyINP, sammyRTO_tot)

            pw_cap_true, par_cap_true = sammyOUT_cap.pw, sammyOUT_cap.par
            pw_tot_true, par_tot_true = sammyOUT_tot.pw, sammyOUT_tot.par

            ### Final Bayes fit parameterized estimate to true
            sammyRTO_cap_bayes = copy.copy(sammyRTO_cap)
            sammyRTO_cap_bayes.solve_bayes = True
            sammyRTO_cap_bayes.recursive = True
            sammyRTO_cap_bayes.recursive_opt['print'] = False
            sammyRTO_cap_bayes.recursive_opt['threshold'] = 1e-7
            sammyRTO_cap_bayes.recursive_opt['iterations'] = 7

            sammyRTO_tot_bayes = copy.copy(sammyRTO_cap_bayes)
            sammyRTO_tot_bayes.reaction = 'total'
            sammyRTO_tot_bayes.inptemplate = 'allexptot_1sg.inp'

            exp_df_cap = pd.DataFrame({"E":E_cap, "exp_trans":dc, "exp_trans_unc":dc_unc})
            exp_df_tot = pd.DataFrame({"E":E_tot, "exp_trans":dt, "exp_trans_unc":dt_unc})

            sammyINP = sammy_classes.SammyInputData(particle_pair       = Ta_pair, 
                                                    resonance_ladder    = par_ladder,
                                                    experimental_data   = exp_df_cap,
                                                    initial_parameter_uncertainty = 0.1,
                                                    temp = 304.5, FP=35.185, frac_res_FP=0.049600, target_thickness=0.067166)
            sammyOUT_capfit = sammy_functions.run_sammy(sammyINP, sammyRTO_cap_bayes)
            pw_cap_fit, par_cap_fit = sammyOUT_capfit.pw, sammyOUT_capfit.par_post  
            sammyINP.experimental_data = exp_df_tot
            sammyOUT_totfit = sammy_functions.run_sammy(sammyINP, sammyRTO_tot_bayes)
            pw_tot_fit, par_tot_fit = sammyOUT_totfit.pw, sammyOUT_totfit.par_post
            
            save_Gn_error_cap.append((par_cap_true["Gn1"] - par_cap_fit["Gn1"]).item())
            save_Gg_error_cap.append((par_cap_true["Gg"] - par_cap_fit["Gg"]).item())
            save_cap_residual_cap.append(np.sum(abs((pw_cap_true.theo_xs - pw_cap_fit.theo_xs).values))) # type: ignore

            save_Gn_error_tot.append((par_tot_true["Gn1"] - par_tot_fit["Gn1"]).item())
            save_Gg_error_tot.append((par_tot_true["Gg"] - par_tot_fit["Gg"]).item())
            save_cap_residual_tot.append(np.sum(abs((pw_tot_true.theo_xs - pw_tot_fit.theo_xs).values))) # type: ignore
            
            save_sim_cap.append(sim_cap[iest,itrue])
            save_sim_tot.append(sim_tot[iest,itrue])


        except:
            print(f"FAILED Case e/b:{e}/{b}, iest/itrue:{iest}/{itrue}")


np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_Gn_error_cap.npy', save_Gn_error_cap)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_Gg_error_cap.npy', save_Gg_error_cap)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_cap_residual_cap.npy', save_cap_residual_cap)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_Gn_error_tot.npy', save_Gn_error_tot)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_Gg_error_tot.npy', save_Gg_error_tot)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_cap_residual_tot.npy', save_cap_residual_tot)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_sim_tot.npy', save_sim_tot)
np.save(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/test_GLLSconvergence/iest{iestbase}_sim_cap.npy', save_sim_cap)


