#%%
import numpy as np
import pandas as pd
import os
import sys

import scipy.stats as sts

from models.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.resonance_statistics import make_res_par_avg
from ATARI.theory.scattering_params import FofE_recursive, FofE_explicit
from ATARI.utils.stats import likelihood_ratio_test, likelihood_val, chi2_val
from utils.io.atario import fill_resonance_ladder
import utils.io.atario as io
from ATARI.sammy_interface import sammy_functions
from ATARI.sammy_interface import sammy_functions, sammy_classes

from ATARI.utils.io.experimental_parameters import BuildExperimentalParameters_fromDIRECT, DirectExperimentalParameters
from ATARI.utils.io.theoretical_parameters import BuildTheoreticalParameters_fromHDF5, BuildTheoreticalParameters_fromATARI, DirectTheoreticalParameters
from ATARI.utils.io.pointwise_container import BuildPointwiseContainer_fromHDF5, BuildPointwiseContainer_fromATARI, DirectPointwiseContainer
from ATARI.utils.io.data_container import BuildDataContainer_fromBUILDERS, BuildDataContainer_fromOBJECTS, DirectDataContainer

from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear
from qpsolvers import solve_qp
from scipy.optimize import linprog

sys.path.insert(0, '/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/')

import functions as fn 
import classes as cls
from bisectfns import solve_qp_w_constraint, bisect_and_solve, get_bounding_wcons, get_target_numres_array, get_target_wcon_solw_arrays



# %%


def fit(casenum, case_file, res_par_avg, write_folder):

    ac=0.81271; M=180.948030; m=1; I=3.5; i=0.5; l_max=1      

    spin_groups = [ (3.0,1,0) ]
    average_parameters = {'3.0':res_par_avg}
    Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                    input_options={},
                                    spin_groups=spin_groups,
                                    average_parameters=average_parameters )

    casenum = 9
    ### Build from hdf5
    builder_exppar = BuildExperimentalParameters_fromDIRECT(0.067166, 0, 1e-2)
    exppar = builder_exppar.construct()

    builder_theopar = BuildTheoreticalParameters_fromHDF5('true', case_file, casenum, Ta_pair)
    truepar = builder_theopar.construct()

    builder_pw = BuildPointwiseContainer_fromHDF5(case_file, casenum)
    pw = builder_pw.construct_lite_w_CovT()

    builder_dc = BuildDataContainer_fromOBJECTS( pw, exppar, [truepar])
    dc = builder_dc.construct()
    dc.pw.fill_exp_xs(dc.experimental_parameters)

    #%% Initial feature reduction with LP then unconstrained problem

    Elam_features, Gtot_features, Gn = fn.get_parameter_grid_new(dc.pw.exp.E, average_parameters['3.0'], num_Er=int(2e2), num_Gt=int(5))
    w_threshold = 1e-6
    prob = cls.ProblemHandler(w_threshold)

    fb0 = prob.get_FeatureBank(dc, Elam_features, Gtot_features)
    inp0 = prob.get_MatrixInputs(dc, fb0)

    sol_lp0 = cls.Solvers.solve_linear_program(inp0)
    print(fb0.nfeatures)
    print(np.count_nonzero(sol_lp0>0))

    qpopt = cls.QPopt(verbose=True,
                    abstol = 1e-6,
                    reltol = 1e-6,
                    feastol=1e-6,
                        maxiters = 200)

    fb1, sol_lp0_ereduced = prob.reduce_FeatureBank(fb0, sol_lp0)
    inp1 = prob.get_MatrixInputs(dc, fb1)
    fb1.solution_ws = cls.Solvers.solve_quadratic_program(inp1, qpopt)

    #%% # Run bisection routine

    # determine mins and maxes
    min_wcon = prob.get_MinSolvableWeight(fb1.nfeatures, inp1)
    max_wcon = np.sum(fb1.solution_ws)
    max_numres = np.count_nonzero(fb1.solution_ws>prob.w_threshold)
    try:
        min_wcon_solw = solve_qp_w_constraint(prob, inp1, min_wcon*1.000, qpopt)
        min_numres = np.count_nonzero(min_wcon_solw>prob.w_threshold)
    except:
        min_wcon_solw = solve_qp_w_constraint(prob, inp1, min_wcon*1.001, qpopt)
        min_numres = np.count_nonzero(min_wcon_solw>prob.w_threshold)

    print(f'Minimum Resonance Solution: {min_numres}')
    print(f'Maximum Resonance Solution: {max_numres}')

    # determine targets
    target_numres = get_target_numres_array(25, max_numres, min_numres)
    target_wcon, target_sol_ws = get_target_wcon_solw_arrays(target_numres, min_wcon, min_wcon_solw,max_numres, max_wcon, fb1.solution_ws, target_wcon=None, sol_ws=None)
    save_all = [(min_wcon, min_numres), (max_wcon, max_numres)]

    print('Running bisection search')
    qpopt.verbose=False
    # Run bisection routine
    for current_target_ires in target_numres[::-1]:
        print(f'Found: {target_numres[target_wcon!=0]}')
        print(f'Current target: {current_target_ires}')
        minwcon, maxwcon = get_bounding_wcons(current_target_ires, save_all)
        target_wcon, target_sol_ws, save_all, _ = bisect_and_solve(inp1, prob, qpopt, current_target_ires,
                                                                    minwcon, maxwcon,
                                                                    target_numres, target_wcon, target_sol_ws, save_all)

    # %%  Step 3, Solve reduced, unconstrained solution for each integer number of resonances

    print("Solving unconstrained, reduced problem for each target number of resonances.")

    integer_feature_solutions = {key: cls.FeatureBank for key in target_numres[target_wcon!=0]}
    for numres in target_numres[target_wcon!=0]:
        inumres = numres-min(target_numres)
        # constrained unreduced
        constrained_solution = target_sol_ws[:, inumres]

        fb3, solw_reduced = prob.reduce_FeatureBank(fb1, constrained_solution)
        inp3 = prob.get_MatrixInputs(dc, fb3)
        fb3.solution_ws = cls.Solvers.solve_quadratic_program(inp3, qpopt)

        integer_feature_solutions[numres] = fb3

    # %%  eliminate smaller resonanaces within <Q01 <D> of another resonanace

    print('Eliminating resonances within Q01 <D> of a larger resonanace')

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

    print(f'Surviving integer number of resonance solutions: {list(integer_resonance_solutions.keys())}')

    # %% Step 4, run GLLS on transmission with reduced, unconstrained solution from 3 as prior

    def run_sammy_return_full_ladder(sammyINP, sammyRTO):
        pw_posterior, par_posterior = sammy_functions.run_sammy(sammyINP, sammyRTO)
        par_posterior.rename(columns={'Gn1':'Gn'}, inplace=True)
        par_posterior = fill_resonance_ladder(par_posterior, Ta_pair, J=3.0,
                                                        chs=1.0,
                                                        lwave=0.0,
                                                        J_ID= 1.0  )
        return pw_posterior, par_posterior


    def run_recursive_sammy(sammyINP, sammyRTO, pw_prior, exp_df, CovT, Dchi2_threshold = 0.1, iterations = 25):
        Dchi2 = 100; itter = 0
        pw_posterior_new = None
        par_posterior_new = sammyINP.resonance_ladder

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

    print("Running GLLS from feature bank solution")
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
        lst, posterior = run_recursive_sammy(sammyINP, sammyRTO, dc.pw.exp[f'{numres}_prior_trans'], dc.pw.exp, dc.pw.CovT, Dchi2_threshold=0.01, iterations=3)
        posterior = fill_resonance_ladder(posterior, Ta_pair, J=3.0, chs=1, lwave=0.0, J_ID=1)
        integer_resonance_solutions[numres]['posterior'] = posterior

        est_par_builder = BuildTheoreticalParameters_fromATARI(f'{numres}_post', posterior, Ta_pair)
        est_par = est_par_builder.construct()
        dc.add_theoretical_parameters(est_par)

    dc.models_to_pw()

    # %% Step 5, Likelihood ratio test on each of the posterior solutions to determine which number of resonances we should have

    print('Now running recursive likelihood ratio test')

    posterior_ires_chi2 = [ (int(key.split('_')[0]),
                            chi2_val(dc.pw.exp[f'{key}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT))
                                for key in dc.theoretical_parameters.keys() if key!='true' and key!='final' and key.split('_')[1]=='post']
    posterior_ires_chi2 = np.array(posterior_ires_chi2)
    posterior_ires_chi2 = posterior_ires_chi2[posterior_ires_chi2[:, 0].argsort()]
        
    ### Find first plausible model
    for i in range(len(posterior_ires_chi2)):
        likelihood = likelihood_val( np.array(dc.pw.exp.exp_trans), np.array(dc.pw.exp[f'{int(posterior_ires_chi2[i,0])}_post_trans']), np.array(dc.pw.CovT))

        print(np.log(likelihood))
        if np.log(likelihood) >= -100:
            istart = i
            break
        elif i == len(posterior_ires_chi2)-1:
            istart = 0
            break

    # ### Loop over different sig levels
    siglevels = [0.00001, 0.0001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
    for sig in siglevels:
        print(f"Running LRT for significance level: {sig}")

        inull = istart 
        ialt = inull
        iend = np.shape(posterior_ires_chi2)[0]
        while ialt < iend:
            p_value = 1.0
            while p_value > sig:
                ialt += 1
                if ialt == iend:
                    break
                df_diff = posterior_ires_chi2[ialt][0]*3 - posterior_ires_chi2[inull][0]*3
                lrt_stat, p_value = likelihood_ratio_test(posterior_ires_chi2[inull][1], posterior_ires_chi2[ialt][1], df_diff)
                print(f"Model {posterior_ires_chi2[inull][0]} vs. Model {posterior_ires_chi2[ialt][0]}:\n p={p_value} D={lrt_stat}")
            if ialt == iend:
                selected_model_ires = int(posterior_ires_chi2[inull][0])
                break
            else:
                inull = ialt

        print(f'Model Selected: {posterior_ires_chi2[inull][0]}')

        print("Saving")

        final_estimate = dc.theoretical_parameters[f'{selected_model_ires}_post'].resonance_ladder
        final_estimate.to_csv(os.path.join(write_folder, f'par_est_{casenum}_pv_{sig:0<10f}.csv'))

    return


def main(isample):

    for Gg_DOF_true in [10000]:#d[10,100,1000,10000]:
    
        case_file = f'/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/varyGg/GgDOF_{Gg_DOF_true}.hdf5'
        
        dT_folder = f'./gdT{Gg_DOF_true}'
        if os.path.isdir(os.path.realpath(dT_folder)):
            pass
        else:
            os.mkdir(os.path.realpath(dT_folder))

        for Gg_avg in [90]:#[30, 60, 90]:
            for Gg_dof in [10]:#[10, 100, 1000, 10000]:
                write_folder = os.path.join(dT_folder, f'gaF{Gg_avg}_gdF{Gg_dof}/')

                if os.path.isdir(os.path.realpath(write_folder)):
                    pass
                else:
                    os.mkdir(os.path.realpath(write_folder))
                
                res_par_avg = make_res_par_avg(D_avg = 8.79, 
                                                Gn_avg= 1.617, 
                                                n_dof = 1, 
                                                Gg_avg = Gg_avg, #64.0, 
                                                g_dof = Gg_dof, 
                                                print = False)
                try:
                    fit(isample, case_file, res_par_avg, write_folder)
                except:
                    pass

# args = sys.argv[1]
main(1)

# import time

# start_time = time.time()

# end_time = time.time()
# elapsed_time = end_time - start_time

# with open(f"/home/nwalton1/reg_perf_tests/lasso/E75_125/outfiles/tfit_{isample}_{elapsed_time}", 'w') as f:
#     f.close()

