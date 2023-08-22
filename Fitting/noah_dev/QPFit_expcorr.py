# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import h5py
import scipy.stats as sts

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.xs import SLBW
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat
from ATARI.utils.datacontainer import DataContainer
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.utils.stats import chi2_val
from ATARI.theory.resonance_statistics import make_res_par_avg
import ATARI.utils.atario as io
from ATARI.utils.misc import fine_egrid 

import functions as fn 
# import classes as cls


# %%
# %matplotlib widget

# %% [markdown]
# ### Generate data with syndat module

# %%
ac = 0.81271  # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider

E_min_max = [75, 125]
energy_grid = E_min_max
input_options = {'Add Noise': True,
                'Calculate Covariance': False,
                'Compression Points':[],
                'Grouping Factors':None}

# experiment_parameters = {'bw': {'val':0.0256,   'unc'   :   0},
#                          'n':  {'val':0.067166,     'unc'   :0}}
experiment_parameters = {'bw': {'val':0.1024,   'unc'   :   0},
                         'n':  {'val':0.067166,     'unc'   :0}}

exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)

# for Gg_DOF in [10,50,100,1000,10000]:
Gg_DOF = 1000

from ATARI.theory import scattering_params
shift, penetration, phi, k = scattering_params.FofE_explicit(exp.energy_domain, ac, M, m, 0)
res_par_avg = make_res_par_avg(D_avg = 8.79, 
                            Gn_avg= np.mean(penetration)*2*46.4, 
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

resonance_ladder = Ta_pair.sample_resonance_ladder(energy_grid, spin_groups, average_parameters)

sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'SLBW',
    reaction = 'transmission',
    experimental_corrections = 'all_exp',
    # sammy_runDIR = 'SAMMY_runDIR_1',
    shell = 'zsh'
    )


# %% [markdown]
# #### Generate and build dc from local 

# %%

sammyINP = sammy_classes.SammyInputData(
    particle_pair = Ta_pair,
    resonance_ladder = resonance_ladder,
    energy_grid = exp.energy_domain)

pw_df, par_df = sammy_functions.run_sammy(sammyINP, sammyRTO)
exp.run(pw_df)

# build experimental parameters
builder_exp_par = BuildExperimentalParameters_fromDIRECT(0.067166, 0, 1e-2)
exp_par = builder_exp_par.construct()

# build theoretical parameters
builder_theo_par = BuildTheoreticalParameters_fromATARI('true', resonance_ladder, Ta_pair)
theo_par = builder_theo_par.construct()

# build pointwise data
builder_pw = BuildPointwiseContainer_fromATARI(exp.trans)
pw = builder_pw.construct_lite_w_CovT()
pw.add_model(theo_par, exp_par) 

# build data container
builder_dc = BuildDataContainer_fromOBJECTS(pw, exp_par, [theo_par])
dc = builder_dc.construct()
# dc.pw.add_model(theo_par, exp_par)
dc.pw.fill_exp_xs(dc.experimental_parameters)

# %%
pw_df

# %% [markdown]
# #### generate/build dc from hdf5

# %%
# case_file = f'/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/lasso/test.hdf5'
# dataset_range = (0, 10)
# samples_not_generated = generate(Ta_pair, exp, 
#                                         'sammy', 
#                                         dataset_range, 
#                                         case_file,
#                                         fixed_resonance_ladder=None, 
#                                         open_data=None,
#                                         vary_Erange=None,
#                                         use_hdf5=True,
#                                         overwrite = False, 
#                                         sammy_RTO = sammyRTO)

# casenum = 3

# ### Build from hdf5
# builder_exppar = BuildExperimentalParameters_fromDIRECT(0.067166, 0, 1e-2)
# exppar = builder_exppar.construct()

# builder_theopar = BuildTheoreticalParameters_fromHDF5('true', case_file, casenum, Ta_pair)
# truepar = builder_theopar.construct()

# builder_pw = BuildPointwiseContainer_fromHDF5(case_file, casenum)
# pw = builder_pw.construct_lite_w_CovT()

# builder_dc = BuildDataContainer_fromOBJECTS( pw, exppar, [truepar])
# dc = builder_dc.construct()
# dc.pw.fill_exp_xs(dc.experimental_parameters)


# %%

figure()
# plot(pw.fine.E, pw.fine.theo_xs)
# plot(dc.pw.exp.E, dc.pw.exp.true_xs)
errorbar(dc.pw.exp.E, dc.pw.exp.exp_xs, yerr=dc.pw.exp.exp_xs_unc, fmt='.', capsize=2)
yscale('log')
# plot(dc.pw.exp.E, dc.pw.exp.exp_trans, '.')
# ylim([-max_xs*.1, max_xs*1.25])

# %% [markdown]
# ## Functions

# %%

import numpy as np
import pandas as pd
import scipy.stats as sts

from scipy.linalg import block_diag
from numpy.linalg import inv

from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat

from operator import itemgetter
from itertools import groupby

from classes import FeatureBank

# ================================================================================================
# Functions for setting up the feature bank and constriants
# ================================================================================================

def get_parameter_grid(energy_grid, res_par_avg, num_Er, num_Gt, option=0):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_grid) + res_par_avg['Gt99']/10e3
    min_Elam = min(energy_grid) - res_par_avg['Gt99']/10e3

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
    Gt_list = np.array(Gt_list)*1e-3; num_Gt = len(Gt_list)
    Gn_list = np.array(Gn_list)*1e-3; num_Gn = len(Gn_list)

    Gn_matrix = np.tile(Gn_list, num_Gt*num_Er)
    Gt_matrix = np.tile(np.repeat(Gt_list, num_Gn), num_Er)
    Er_matrix = np.repeat(Er_list, num_Gt*num_Gn)

    resonance_ladder_array = np.column_stack((Er_matrix, Gt_matrix, Gn_matrix))
    return resonance_ladder_array


def get_feature(E, Res, particle_pair, kinematic_constant, phi, lwave, P):
    Elam=Res[0]; Gtot=Res[1]; Gn=Res[2]
    _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    PPElam = P/PElam
    # A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
    #                                         -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
    A_column =  kinematic_constant*Gn* ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                    - (Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
    return A_column
            
def get_feature_bank_from_resonance_ladder(E, particle_pair, resonance_ladder_array, solver, sammy_RTO=None):
    E = np.array(E)
    number_of_resonances = len(resonance_ladder)
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
                sammyINP = sammy_classes.SammyInputData(particle_pair = particle_pair, resonance_ladder =pd.DataFrame([np.append(Res,[3.0,1,0,1])], columns=["E","Gt","Gn","J","chs","lwave","J_ID"], index=[0]), energy_grid = E)
                pw, par = sammy_functions.run_sammy(sammyINP, sammyRTO)
                Resonance_Matrix[:, iRes] = pw.theo_xs-potential_scattering.flatten()
    else:
        raise ValueError(f"Solver: {solver} not recognized")

    return Resonance_Matrix, potential_scattering.flatten()


def get_resonance_ladder_from_feature_bank(weights, Elam_features, Gtot_features, threshold):
    feature_indices = np.argwhere(weights>threshold).flatten()
    resonances = []
    for ifeature in feature_indices:
        Efeature_index, Gfeature_index = divmod(ifeature, len(Gtot_features))
        Elam = Elam_features[Efeature_index]
        Gt = Gtot_features[Gfeature_index]*1e3
        w = weights[ifeature]
        Gnx = Gt*w
        Gg = Gt-Gnx
        resonances.append([Elam, Gt, Gnx, Gg, w])
    resonance_ladder = pd.DataFrame(resonances, columns=['E', 'Gt', 'Gnx', 'Gg', 'w'])
    return resonance_ladder

def get_bound_arrays(nfeat, bounds):
    return np.ones(nfeat)*bounds[0], np.ones(nfeat)*bounds[1]


def get_0Trans_constraint(exp_E, index_0T, max_xs, particle_pair, feature_pairs):
    
    # group consecutive ranges of 0T
    consecutive_ranges = []
    for k,g in groupby(enumerate(index_0T),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        consecutive_ranges.append((group[0],group[-1]))

    # create a fine grid in energy where we have blackout resonances
    fineE_0T = []
    for each_group in consecutive_ranges:
        Emin = exp_E[each_group[0]]
        Emax = exp_E[each_group[1]]
        fineE = np.linspace(Emin, Emax, max(int(abs(Emax-Emin)*5e1),1)  )
        fineE_0T.extend(fineE)
    
    # calculate resonance matrix on fine E grid for constraint
    E = np.array(fineE_0T)
    number_of_resonances = np.shape(feature_pairs)[0]
    Constraint_Matrix = np.zeros((len(E), number_of_resonances))
    Constraint = - np.array([max_xs]*len(E))

    lwave = 0
    _, P, phi, k = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    g = gstat(3.0, particle_pair.I, particle_pair.i)
    kinematic_constant = (4*np.pi*g/k**2)

    for ifeature, resonance in enumerate(feature_pairs):

        Elam = resonance[0]
        Gtot = resonance[1]

        _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
        PPElam = P/PElam
        # A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
        #                                         -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) ) 
        A_column =  kinematic_constant* ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                            - (Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  

        Constraint_Matrix[:, ifeature] = - A_column

    return Constraint_Matrix ,Constraint 



def remove_nan_values(full_xs, full_cov, full_pscat, full_feature_matrix):
    index_0T = np.argwhere(np.isnan(full_xs)).flatten()
    index_finiteT = np.argwhere(np.isfinite(full_xs)).flatten()

    cov = full_cov.copy()[index_finiteT, :]
    cov = cov[:, index_finiteT]

    xs = full_xs[index_finiteT]
    pscat = full_pscat[index_finiteT]

    feature_matrix = full_feature_matrix[index_finiteT, :]

    return xs, cov, pscat, feature_matrix, index_0T


def get_reduced_features(full_feature_matrix, solution_ws, w_threshold, feature_pairs):
    index_w_surviving = np.argwhere(solution_ws>w_threshold).flatten()
    reduced_solw = solution_ws[index_w_surviving]
    reduced_feature_matrix = full_feature_matrix[:, index_w_surviving]
    reduced_feature_pairs = feature_pairs[index_w_surviving,:]
    return reduced_feature_matrix, reduced_feature_pairs, reduced_solw



# %%


# %% [markdown]
# ## Classes

# %%
## Redefine classes
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import lsq_linear
from qpsolvers import solve_qp
from scipy.optimize import linprog
from numpy.linalg import inv
from scipy.linalg import block_diag

import functions as fn
from ATARI.utils.io.data_container import DataContainer


def get_resonance_ladder_from_feature_pairs(weights, feature_pairs):
    Elam = feature_pairs[:,0]
    Gt = feature_pairs[:,1]*1e3
    weights = weights.flatten()
    # Gnx = Gt*weights
    Gn = weights*1e3
    Gg = Gt-Gn
    resonances = np.array([Elam, Gt, Gn, Gg, weights])
    resonance_ladder = pd.DataFrame(resonances.T, columns=['E', 'Gt', 'Gn', 'Gg', 'w'])
    return resonance_ladder

@dataclass
class FeatureBank:
    feature_matrix: np.ndarray
    feature_pairs: np.ndarray
    potential_scattering: np.ndarray
    nfeatures: int
    w_bounds: tuple
    solution_ws: Optional[np.ndarray] = None

    @property
    def model(self):
        return self.feature_matrix@self.solution_ws+self.potential_scattering
    
    def get_parameter_solution(self):
        return get_resonance_ladder_from_feature_pairs(self.solution_ws, self.feature_pairs)

@dataclass
class MatrixInputs:
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    h: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    index_0T: np.ndarray
    
@dataclass
class QPopt:
    solver: str = "cvxopt"
    verbose: bool = False
    abstol: float = 1e-12
    reltol: float = 1e-12
    feastol: float = 1e-8
    maxiters: float =  100


class Solvers:

    @staticmethod
    def solve_quadratic_program(inputs: MatrixInputs, qpopt: QPopt) -> Optional[np.ndarray] :

        sol_w = solve_qp(inputs.P, inputs.q, G=inputs.G, h=inputs.h, A=None, b=None, lb=inputs.lb, ub=inputs.ub, 
                                                                                                    solver=qpopt.solver,
                                                                                                    verbose=qpopt.verbose,
                                                                                                    abstol=qpopt.abstol,
                                                                                                    reltol=qpopt.reltol,
                                                                                                    feastol=qpopt.feastol,
                                                                                                    maxiters=qpopt.maxiters
                                                                                                    )
                                                                                                
        return sol_w.reshape(-1,1)
    
    @staticmethod
    def solve_linear_program(inputs: MatrixInputs):
        # if inputs.ub is None:
        #     ub = np.ones_like(inputs.lb)*1000
        # else:
        #     ub = inputs.ub
        sol_w = linprog(inputs.q, A_ub=inputs.G, b_ub=inputs.h, bounds=np.array([inputs.lb, inputs.ub]).T)
        return sol_w.x
        



# %%


# %%
# Er_list, Gt_list, Gn_list = get_parameter_grid(energy_grid, res_par_avg, num_Er, num_Gt, option=0)
# resonance_ladder_array = get_resonance_ladder_from_parameter_grid(Er_list, Gt_list, Gn_list)

resonance_bank, pscat = get_feature_bank_from_resonance_ladder(dc.pw.exp.E, 
                                                            dc.theoretical_parameters['true'].particle_pair, 
                                                            resonance_ladder[["E","Gt","Gn"]].values,
                                                             "sammy", sammy_RTO=sammyRTO)


fb0 = FeatureBank(resonance_bank, resonance_ladder[["E","Gt","Gn"]].values, pscat.reshape((-1,1)), len(resonance_ladder[["E","Gt","Gn"]].values), [0,res_par_avg[""]] )


# %%


# %%
### Unit test for feature bank construction

fb0 = prob.get_FeatureBank(dc, resonance_ladder.E.values, resonance_ladder.Gt.values)

# inp0 = prob.get_MatrixInputs(dc, fb0)
fb0.solution_ws = np.atleast_2d(resonance_ladder.Gn.values).T
# test_par = TheoreticalParameters(Ta_pair, pd.DataFrame({'E':575, 'Gt':100, 'Gn':50, 'Gg':50, 'J':3.0,'chs':1,'lwave':0.0, 'J_ID':None}, index=[0]))
# test_par = TheoreticalParameters(Ta_pair, resonance_ladder)
builder_theo_par = BuildTheoreticalParameters_fromATARI('test', resonance_ladder, Ta_pair)
test_par = builder_theo_par.construct()

test,_,_ = SLBW(dc.pw.exp.E, Ta_pair, test_par.resonance_ladder)

figure()
plot(dc.pw.exp.E, fb0.model-test)
plot(dc.pw.exp.E, test)


# %% [markdown]
# ## Step 0, reduce initial feature bank for computational speed

# %%
### Step 0

# Elam_features, Gtot_features = fn.get_parameter_grid(dc.pw.exp.E, average_parameters, '3.0', 2e-1, 3e0)
Elam_features, Gtot_features, Gn = fn.get_parameter_grid_new(dc.pw.exp.E, average_parameters['3.0'], num_Er=int(2e2), num_Gt=int(5))
Gtot_features = np.append(Gtot_features, np.round(np.array(dc.theoretical_parameters['true'].resonance_ladder.Gt),1)*1e-3 )
Elam_features = np.append(Elam_features, np.round(np.array(dc.theoretical_parameters['true'].resonance_ladder.E),1))
# Elam_features = np.round(np.array(theo_resladder.E),1)
# Gtot_features = np.array(theo_resladder.Gt)*1e-3
# Elam_features = np.array(theo_resladder.E)

w_threshold = 1e-6
prob = cls.ProblemHandler(w_threshold)

fb0 = prob.get_FeatureBank(dc, Elam_features, Gtot_features)
inp0 = prob.get_MatrixInputs(dc, fb0)

sol_lp0 = cls.Solvers.solve_linear_program(inp0)
print(fb0.nfeatures)
print(np.count_nonzero(sol_lp0>0))
Gtot_features

# %% [markdown]
# ## Step 1, solve unconstrained problem

# %%
qpopt = cls.QPopt(verbose=True,
                abstol = 1e-6,
                reltol = 1e-6,
                feastol=1e-6,
                    maxiters = 200)

# %%
### Step 1
fb1, sol_lp0_ereduced = prob.reduce_FeatureBank(fb0, sol_lp0)
inp1 = prob.get_MatrixInputs(dc, fb1)
# fb1 = fb0
# inp1 = inp0
fb1.solution_ws = cls.Solvers.solve_quadratic_program(inp1, qpopt)


# %%


# %%
# integer_feature_solutions[numres]

figure(figsize=(5,4))
# plot(dc.pw.exp.E, dc.pw.exp.theo_xs, 'g')
errorbar(dc.pw.exp.E, dc.pw.exp.exp_xs, yerr=dc.pw.exp.exp_xs_unc, fmt='k.',alpha=0.2, capsize=2,elinewidth=0.5, ms=2, zorder=0, label='Data')
plot(dc.pw.exp.E, fb1.model, 'r' ,lw=1.75, zorder=2, label="Fit")
yscale('log')
ylabel('Total Cross Section')
xlabel("Energy (eV)")
title("Initial Fit")
legend()
ylim([1e0, 3e1])
tight_layout()


# for each in models:
#     figure(figsize=(5,4))
#     errorbar(dc.pw.exp.E, dc.pw.exp.exp_xs, yerr=dc.pw.exp.exp_xs_unc, fmt='k.', alpha=0.2,capsize=2,elinewidth=0.5, ms=2, zorder=0, label='Data')
#     plot(dc.pw.exp.E, each, 'r',lw=1.75, zorder=2, label="Fit")
#     yscale('log')
#     ylabel('Total Cross Section')
#     xlabel("Energy (eV)")
#     title("Inducing Sparsity")
#     legend()
#     ylim([1e0, 3e1])
#     tight_layout()
# # bins = hist(np.log(fb1.solution_ws), bins=50)
# np.count_nonzero(fb1.solution_ws>prob.w_threshold)

# %%
# models = []
for i in [0.22]:#[0.8, 0.6, 0.4, 0.3, 0.25]:
    solw = solve_qp_w_constraint(inp1, max_wcon*i, qpopt)
    fb1.solution_ws = solw
    models.append(fb1.model)


# %%
figure(figsize=(5,4))
errorbar(dc.pw.exp.E, dc.pw.exp.exp_xs, yerr=dc.pw.exp.exp_xs_unc, fmt='k.', alpha=0.2,capsize=2,elinewidth=0.5, ms=2, zorder=0, label='Data')
plot(dc.pw.exp.E, fb1.model, 'r',lw=1.5, zorder=2, label="Fit")
yscale('log')
ylabel('Total Cross Section')
xlabel("Energy (eV)")
title("Inducing Sparsity")
legend()
ylim([1e0, 3e1])
tight_layout()

# %%
def solve_qp_w_constraint(inp_uncon, wcon, qpopt: cls.QPopt):
    inp_con = prob.get_ConstrainedMatrixInputs(inp_uncon, wcon)
    sol = cls.Solvers.solve_quadratic_program(inp_con, qpopt)
    return sol

# %% [markdown]
# ## Step 2, run bisection routine

# %%
### Bisection Routine
def bisect(x0,x1):
    return (x0+x1)/2

def bisect_and_solve(current_target_ires, minval, maxval, target_numres, wcon, sol_ws, save_all, searching=True, termination_threshold=1e-3):

    icurrent_target_ires = current_target_ires-min(target_numres)
    
    while searching:

        # check termination criteria
        if abs((minval-maxval)/maxval) < termination_threshold:
            searching = False
        elif wcon[icurrent_target_ires] > 0:
            searching = False

        # bisect, solve, and save result
        new_w = bisect(minval, maxval)
        con_sol_ws = solve_qp_w_constraint(inp1, new_w, qpopt) # TODO: Make dc and fb dynamic here!
        if con_sol_ws is None:
            wcon, sol_ws, save_all, searching = bisect_and_solve(current_target_ires, new_w, maxval, target_numres, wcon, sol_ws, save_all, searching=searching)
        else:
            numres_sol = np.count_nonzero(con_sol_ws>w_threshold)
            ires_sol = numres_sol-min(target_numres)
            save_all.append((new_w, numres_sol))

            if numres_sol == current_target_ires:
                #record if current target
                if wcon[ires_sol] < new_w:
                    wcon[ires_sol] = new_w
                    sol_ws[:,ires_sol] = con_sol_ws.flatten()

            elif numres_sol > current_target_ires:
                # record other if in target numres
                if numres_sol in target_numres:
                    if wcon[ires_sol] < new_w:
                        wcon[ires_sol] = new_w
                        sol_ws[:,ires_sol] = con_sol_ws.flatten()
                # bisect again to find current target
                wcon, sol_ws, save_all, searching = bisect_and_solve(current_target_ires, minval, new_w, target_numres, wcon, sol_ws, save_all, searching=searching)

            elif numres_sol < current_target_ires:
                # record other if in target numres
                if numres_sol in target_numres:
                    if wcon[ires_sol] < new_w:
                        wcon[ires_sol] = new_w
                        sol_ws[:,ires_sol] = con_sol_ws.flatten()
                # bisect again to find current target
                wcon, sol_ws, save_all, searching = bisect_and_solve(current_target_ires, new_w, maxval, target_numres, wcon, sol_ws, save_all, searching=searching)

    return wcon, sol_ws, save_all, searching

def get_bounding_wcons(ires_target, save_all):
    temp = np.array(save_all)
    temp = temp[temp[:,0].argsort()]
    index = np.searchsorted(temp[:,1], ires_target, side='left')
    return temp[index-1,0], temp[index,0]

def get_target_numres_array(target_maxres, max_numres, min_numres):
    return np.arange(min_numres, np.min([target_maxres,max_numres])+1, 1)


def get_target_wcon_solw_arrays(target_numres, min_wcon, min_wcon_solw, max_numres, max_wcon, max_solw, target_wcon=None, sol_ws = None):
    if target_wcon is None and sol_ws is None:
        # init targer w constraints and solution vectors
        target_wcon = np.zeros(len(target_numres))
        sol_ws = np.zeros((min_wcon_solw.shape[0], len(target_numres)))
        # add min wconstraint and solution
        if min_wcon == 0:
            min_wcon = 1e-10
        target_wcon[0] = min_wcon
        sol_ws[:, 0] = min_wcon_solw.flatten()
        # add max if max numres (unconstrained solve) is in the target
        if max_numres in target_numres:
            target_wcon[-1] = max_wcon
            sol_ws[:, -1] = max_solw.flatten()
    
    elif target_wcon is None or sol_ws is None:
        raise ValueError("Either target_wcon or sol_ws is None while the other is populated, check input.")
    else:
        elements_to_add = len(target_numres)-len(target_wcon)
        assert elements_to_add >= 0, "target_numres is shorter than target_wcon"
        target_wcon = np.append(target_wcon, [0]*elements_to_add)
        sol_ws = np.append(sol_ws, np.zeros((np.shape(sol_ws)[0], elements_to_add)), axis=1)
    return target_wcon, sol_ws


# %%
### Step 2

w_threshold = 1e-6
# determine mins and maxes
min_wcon = prob.get_MinSolvableWeight(fb1.nfeatures, inp1)
max_wcon = np.sum(fb1.solution_ws)
max_numres = np.count_nonzero(fb1.solution_ws>prob.w_threshold)
min_wcon_solw = solve_qp_w_constraint(inp1, min_wcon*1.000, qpopt)
min_numres = np.count_nonzero(min_wcon_solw>prob.w_threshold)


# solve_qp_w_constraint(inp1, min_wcon*1.001, qpopt)
print(min_numres)
print(max_numres)

# %%
# determine targets
target_numres = get_target_numres_array(50, max_numres, min_numres)
target_wcon, target_sol_ws = get_target_wcon_solw_arrays(target_numres, min_wcon, min_wcon_solw,max_numres, max_wcon, fb1.solution_ws, target_wcon=None, sol_ws=None)
save_all = [(min_wcon, min_numres), (max_wcon, max_numres)]

print(target_numres)
print(target_wcon)

qpopt.verbose=False
# Run bisection routine
for current_target_ires in target_numres[::-1]:
    print(f'Found: {target_numres[target_wcon!=0]}')
    print(f'Current target: {current_target_ires}')
    minwcon, maxwcon = get_bounding_wcons(current_target_ires, save_all)
    target_wcon, target_sol_ws, save_all, _ = bisect_and_solve(current_target_ires,
                                                                minwcon, maxwcon,
                                                                target_numres, target_wcon, target_sol_ws, save_all)
# target_numres = target_numres.append(max_numres)
# target_wcon = target_wcon.append(fb1.solution_ws)
print(target_numres)
print(target_wcon)

# %% [markdown]
# ## Step 3, Solve reduced, unconstrained solution for each integer number of resonances

# %%
integer_feature_solutions = {key: cls.FeatureBank for key in target_numres[target_wcon!=0]}

chi2pairs = []

for numres in target_numres[target_wcon!=0]:
    
    inumres = numres-min(target_numres)

    # constrained unreduced
    constrained_solution = target_sol_ws[:, inumres]

    fb3, solw_reduced = prob.reduce_FeatureBank(fb1, constrained_solution)
    inp3 = prob.get_MatrixInputs(dc, fb3)
    fb3.solution_ws = cls.Solvers.solve_quadratic_program(inp3, qpopt)

    integer_feature_solutions[numres] = fb3

    # chi2 = chi2_val(fb3.model, dc.pw.exp.exp_xs, dc.pw.CovXS)
    # chi2pairs.append((numres, chi2))

    Pchi2 = (1/2)* fb3.solution_ws.T@inp3.P@fb3.solution_ws + inp3.q.T@fb3.solution_ws
    chi2pairs.append((numres, Pchi2))
    
chi2pairs

# %% [markdown]
# ### How should I combine widths and weights if I have resonances very close to one another?
# 
# You cannot exactly calculate a w3 from (w1, w2) at the same energy location. I would like to approximate the combination of resonances that have very similar Elambda locations. The objective for the combination is to minimize the L2 norm between SLBW(w3) and SLBW(w2)+SLBW(w1) where w3 is some function of w2 and w1.

# %%

### Here's where I handle resonances at the same energies and those with weights very small
integer_resonance_solutions = {}
for numres in target_numres[target_wcon!=0]:
    ires_featurebank = integer_feature_solutions[numres]
    ires_resladder = ires_featurebank.get_parameter_solution()

    # drop resonances with weights below threshold
    ires_resladder.drop(ires_resladder[ires_resladder['w'] < prob.w_threshold].index, inplace=True)
    ires_resladder.reset_index(inplace=True, drop=True)
    
    # drop smaller resonances within Q01 spacing from one-another
    index = np.argwhere(np.diff(ires_resladder.E.values) < res_par_avg['D01'])
    idrop = []
    for ires in index.flatten():
        local_index = [ires, ires+1]
        smaller_res = np.argmin(ires_resladder.w.values[local_index])
        idrop.append(local_index[smaller_res])
    ires_resladder.drop(idrop, inplace=True)
    ires_resladder.reset_index(inplace=True, drop=True)


    new_numres = len(ires_resladder)
    ires_resladder = fill_resonance_ladder(ires_resladder, Ta_pair, J=3.0, chs=1, lwave=0.0, J_ID=1)
    integer_resonance_solutions[new_numres] = {'prior':ires_resladder}

    # add prior to dc
    est_par_builder = BuildTheoreticalParameters_fromATARI(f'{new_numres}_prior', ires_resladder, Ta_pair)
    est_par = est_par_builder.construct()
    dc.add_theoretical_parameters(est_par)

integer_resonance_solutions.keys()

# %%
np.any(np.isnan(integer_resonance_solutions[7]['prior']))

# %%
# numres = 4
# numres_other = 7
# rxn = 'trans'

# figure()
# # plot(dc.pw.exp.E, dc.pw.exp[f'theo_{rxn}'], 'g', lw=5)
# plot(dc.pw.exp.E, dc.pw.exp[f'exp_{rxn}'], '.k')
# plot(dc.pw.exp.E, dc.pw.exp[f'{numres_other}_prior_{rxn}'], 'b', lw=2)
# plot(dc.pw.exp.E, dc.pw.exp[f'{numres}_prior_{rxn}'], 'r', lw=1)
# # plot(dc.pw.exp.E, integer_feature_solutions[numres_other].model, 'b', lw=2)
# # plot(dc.pw.exp.E, integer_feature_solutions[numres].model, 'r', lw=1)
# # ylim([-0.1, dc.exp_par.max_xs])

# # # print(chi2_val(dc.pw.exp[f'{numres}_prior_{rxn}'], dc.pw.exp[f'exp_{rxn}'], dc.pw.CovT))

# %%


# %% [markdown]
# ## Step 4, run GLLS on transmission with reduced, unconstrained solution from 3 as prior

# %%
from ATARI.sammy_interface import sammy_functions, sammy_classes
sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'SLBW',
    reaction = 'transmission',
    solve_bayes = True,
    experimental_corrections = 'no_exp',
    one_spingroup = True,
    energy_window = None,
    sammy_runDIR = 'SAMMY_runDIR_1',
    keep_runDIR = False,
    shell = 'zsh'
    )

# %%
integer_resonance_solutions.keys()

# %%

from ATARI.utils.stats import chi2_val
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.sammy_interface import sammy_functions

def run_sammy_return_full_ladder(sammyINP, sammyRTO):
    pw_posterior, par_posterior = sammy_functions.run_sammy(sammyINP, sammyRTO)
    par_posterior.rename(columns={'Gn1':'Gn'}, inplace=True)
    par_posterior = fill_resonance_ladder(par_posterior, Ta_pair, J=3.0,
                                                    chs=1.0,
                                                    lwave=0.0,
                                                    J_ID= 1.0  )

    return pw_posterior, par_posterior


def run_recursive_sammy(sammyINP, sammyRTO, pw_prior, exp_df, CovT, Dchi2_threshold = 0.1, iterations = 25):
   
    Dchi2 = 100
    pw_posterior_new = None
    par_posterior_new = sammyINP.resonance_ladder
    itter = 0
    while Dchi2 > Dchi2_threshold:
        itter += 1
        if itter > iterations:
            break
        
        pw_posterior = pw_posterior_new
        par_posterior = par_posterior_new
        sammyINP.resonance_ladder = par_posterior
        pw_posterior_new, par_posterior_new = run_sammy_return_full_ladder(sammyINP, sammyRTO)

        [df.sort_values('E', axis=0, ascending=True, inplace=True) for df in [pw_posterior_new, exp_df]]
        [df.reset_index(drop=True, inplace=True) for df in [pw_posterior_new, exp_df]]
        CovT.sort_index(axis='index', inplace=True)
        CovT.sort_index(axis='columns', inplace=True)

        chi2_prior = chi2_val(pw_posterior_new.theo_trans, exp_df.exp_trans, CovT)
        chi2_posterior = chi2_val(pw_posterior_new.theo_trans_bayes, exp_df.exp_trans, CovT)
        Dchi2 = chi2_prior - chi2_posterior

    return pw_posterior, par_posterior

# %%
### Run GLLS
for numres in integer_resonance_solutions.keys():
    if numres == 0:
        continue
    prior = integer_resonance_solutions[numres]['prior']
    
    sammyINP = sammy_classes.SammyInputData(
        particle_pair = Ta_pair,
        resonance_ladder = prior, 
        experimental_data = dc.pw.exp, 
        experimental_cov = dc.pw.CovT, 
        initial_parameter_uncertainty = 0.2
    )
    print(f'Running recursive sammy for case {numres}')
    lst, posterior = run_recursive_sammy(sammyINP, sammyRTO, dc.pw.exp[f'{numres}_prior_trans'], dc.pw.exp, dc.pw.CovT, Dchi2_threshold=0.01, iterations=3)
    # lst, posterior = sammy_functions.run_sammy(sammyINP, sammyRTO)

    # posterior.rename(columns={"Gn1":"Gnx"},inplace=True)
    ## if posterior is worse than prior, re-run
    posterior = fill_resonance_ladder(posterior, Ta_pair, J=3.0, chs=1, lwave=0.0, J_ID=1)
    integer_resonance_solutions[numres]['posterior'] = posterior

    est_par_builder = BuildTheoreticalParameters_fromATARI(f'{numres}_post', posterior, Ta_pair)
    est_par = est_par_builder.construct()
    dc.add_theoretical_parameters(est_par)

dc.models_to_pw()

# %%
# dc.theoretical_parameters['3_post'].resonance_ladder #.keys()

figure()
plot(dc.pw.exp.E, dc.pw.exp['true_trans'], 'g')
plot(dc.pw.exp.E, dc.pw.exp['8_prior_trans'], 'b')
# plot(dc.pw.exp.E, dc.pw.exp['6_post_trans'], 'r')
plot(dc.pw.exp.E, dc.pw.exp.exp_trans, '.k')

# %%
### Calculate Chi2 on trans
# [print(key) for key in dc.theoretical_parameters.keys()]
[ (chi2_val(dc.pw.exp[f'{numres}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT), numres) for numres in dc.theoretical_parameters.keys()]

# %% [markdown]
# ## Step 5, Likelihood ratio test on each of the posterior solutions to determine which number of resonances we should have

# %%
# [ (chi2_val(dc.pw.exp[f'{numres}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT), numres) for numres in dc.est_par.keys()]

posterior_ires_chi2 = [ (int(key.split('_')[0]),
                          chi2_val(dc.pw.exp[f'{key}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT))
                              for key in dc.theoretical_parameters.keys() if key!='true' and key!='final' and key.split('_')[1]=='post']
posterior_ires_chi2 = np.array(posterior_ires_chi2)
posterior_ires_chi2 = posterior_ires_chi2[posterior_ires_chi2[:, 0].argsort()]
posterior_ires_chi2

# %%
for key in dc.theoretical_parameters.keys():
    if key!='true' and key!='final' and key.split('_')[1]=='post':
        print(key)
        print(chi2_val(dc.pw.exp[f'{key}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT))

# %%
import numpy as np
from scipy.stats import chi2

def likelihood_ratio_test(X2_null, X2_alt, df):
    """
    Perform a likelihood ratio test for nested models.

    Args:
        LLmin: Log-likelihood of the null (restricted) model.
        LLmax: Log-likelihood of the alternative (unrestricted) model.
        df: Degrees of freedom difference between the two models.

    Returns:
        lrt_stat: Likelihood ratio test statistic.
        p_value: p-value associated with the test statistic.
    """
    # lrt_stat = 2 * (LLalt - LLnull)
    lrt_stat = X2_null - X2_alt
    p_value = 1 - chi2.cdf(lrt_stat, df)
    return lrt_stat, p_value

def likelihood_val(fit):
    return sts.multivariate_normal.pdf( np.array(dc.pw.exp.exp_trans), fit, np.array(dc.pw.CovT) )
    

### Find first plausible model
for i in range(len(posterior_ires_chi2)):
    likelihood = likelihood_val( np.array(dc.pw.exp[f'{int(posterior_ires_chi2[i,0])}_post_trans']) )

    print(np.log(likelihood))
    if np.log(likelihood) >= -100:
        istart = i
        break
    elif i == len(posterior_ires_chi2)-1:
        istart = 0
        break




inull = istart 
ialt = inull
iend = np.shape(posterior_ires_chi2)[0]

significance_level = 0.0001

while ialt < iend:

    # reset p_value
    p_value = 1.0

    while p_value > significance_level:

        ialt += 1
        if ialt == iend:
            break
        df_diff = posterior_ires_chi2[ialt][0]*3 - posterior_ires_chi2[inull][0]*3
        lrt_stat, p_value = likelihood_ratio_test(posterior_ires_chi2[inull][1], posterior_ires_chi2[ialt][1], df_diff)
        print(f"Model {posterior_ires_chi2[inull][0]} vs. Model {posterior_ires_chi2[ialt][0]}:\n p={p_value} D={lrt_stat}")
        # print(f"D: {lrt_stat}")
        # print(f"P-value: {p_value}")

    if ialt == iend:
        selected_model_ires = int(posterior_ires_chi2[inull][0])
        break
    else:
        inull = ialt

print(f'Model Selected: {posterior_ires_chi2[inull][0]}')

# %%
# iresnull = 2
# iresalt = 5

# rxn = 'trans'
# null = dc.pw.exp[f'{iresnull}_post_{rxn}']
# alt = dc.pw.exp[f'{iresalt}_post_{rxn}']

# dat =dc.pw.exp[f'exp_{rxn}']
# figure()
# plot(dc.pw.exp.E, dc.pw.exp[f'true_{rxn}'], 'g', lw=3)
# plot(dc.pw.exp.E, dat, '.k')
# plot(dc.pw.exp.E, null, 'b', lw=1)
# plot(dc.pw.exp.E, alt, 'r', lw=1)


# print(chi2_val(null, dat, dc.pw.CovT))
# print(chi2_val(alt, dat, dc.pw.CovT))

# for each in dc.theoretical_parameters[f'{iresnull}_post'].resonance_ladder.E:
#     axvline(each, ymin=0.93,ymax=1, alpha=1, color='b', lw=3)

# for each in dc.theoretical_parameters[f'{iresalt}_post'].resonance_ladder.E:
#     axvline(each, ymin=0.93,ymax=1, alpha=1, color='r', lw=1)


# %%
sig = 1e-5
Gg_DOF_true = 100
f'./gdT{Gg_DOF_true}_'

# %%


# %%
final_estimate = dc.theoretical_parameters[f'{selected_model_ires}_post'].resonance_ladder

# import time
# start_time = time.time()
# # final_estimate = main(args)
# end_time = time.time()
# elapsed_time = end_time - start_time
# final_estimate['tfit'] = np.ones(len(final_estimate))*elapsed_time
# final_estimate
# final_estimate.to_csv(f'./par_est_{casenum}.csv')


final_par = BuildTheoreticalParameters_fromATARI('final', final_estimate, Ta_pair)
dc.add_theoretical_parameters(final_par)
dc.mem_2_full()
dc.models_to_pw()

# %%
# print(dc.theoretical_parameters[f'{selected_model_ires}_prior'].resonance_ladder)
# print(dc.theoretical_parameters[f'{selected_model_ires}_post'].resonance_ladder)

# print(dc.theoretical_parameters['true'].resonance_ladder)


# %%
est_chi_square = (dc.pw.exp.exp_trans-dc.pw.exp.final_trans) @ inv(dc.pw.CovT) @ (dc.pw.exp.exp_trans-dc.pw.exp.final_trans).T
sol_chi_square = (dc.pw.exp.exp_trans-dc.pw.exp.true_trans) @ inv(dc.pw.CovT) @ (dc.pw.exp.exp_trans-dc.pw.exp.true_trans).T
from scipy import integrate
est_sol_MSE = integrate.trapezoid((dc.pw.fine.true_xs-dc.pw.fine.final_xs)**2, dc.pw.fine.E)

print(f'Chi2 estimate to data: {est_chi_square}')
print(f'Chi2 true to data: {sol_chi_square}')
print(f'Integral SE estimate to true: {est_sol_MSE}')


figure()
plot(dc.pw.fine.E, dc.pw.fine.true_xs, 'g')
plot(dc.pw.fine.E, dc.pw.fine.final_xs, 'r')
yscale('log')
# ylim([0,dc.exp_par.max_xs+100])
figure()
plot(dc.pw.exp.E, dc.pw.exp.true_trans, 'g')
plot(dc.pw.exp.E, dc.pw.exp.final_trans, 'r')

# plot(dc.pw.exp.E, dc.pw.exp[f'{selected_model_ires}_post'], 'b')

plot(dc.pw.exp.E, dc.pw.exp.exp_trans, '.k')


# %% [markdown]
# ## Notes
# 
# #### Initial uncertainty on parameters in sammy
# should be large, we don't have a prior we just want the GLLS to converge.
# 
# Fudge factor cant be too larger for numerics:
#  - M' = (M^-1 + (G.T @ V^-1 @ G.T) )^-1
#  - first term is full rank, second is not guranteed (it is if NEpts>=NPar). Becasue the first term is full rank, then adding it to the second makes the whole thing full rank and therefore invertable. However, if the first term (M^-1 or 1/prior variance) is driven numerically to 0, then you are left with only the second term. If it is not full rank it won't be invertable then the thing will not be numerically stable. 
# 
# #### Sammy is not always converging
# - could run sammy multiple times?
# - also, the problem is resolved when I 'cheat' with the resonance energies. I think having multiple small resonances right on top of one another is causing sammy to run away.
# 
# 
# #### Unexpected behavior when finding integer number of resonances
# Often times the weight constrained solution for an integer number of resonances will have selected a smaller resonance than expected when there's a larger existing resonance that would drive the chi2 even lower. I believe this is a problem with not finding the maximum weight solution to an integer number of resonances.
# 
# - Now I am driving neutron width to the total leaving 0 Gt, currently hard coded max Gn = Gt-0.005
# 
# #### LRT
# I am currently calculating the likelihood using sts.multivariate_normal.pdf. When trying to calculate it using just the chi2 to the data, I get an error because the determinate of the covariance matrix is 0... This seems incorrect.
# 
# Using LRT for nested models only works if the nested parameter are the exact same. I don't beleive that this is the case.
# 
# #### Other
# - Can cutoff weights at 1e-5 or 6 because this corresponds directly to a neutron width this small of a resonance is insignificant

# %%



# %%



