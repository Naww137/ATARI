# %%
import numpy as np
import pandas as pd
import os
import scipy.stats as sts

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.utils.datacontainer import DataContainer
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.utils.stats import chi2_val
from numpy.linalg import inv

import functions as fn 


def main(casenum):

    #%% Import data

    # casenum = 1
    filepath = f"/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125/sample_{casenum}"
    theo_resladder = pd.read_csv(os.path.join(filepath, 'ladder.csv'), index_col=0)
    exp_cov = pd.read_csv(os.path.join(filepath, 'cov.csv'), index_col=0)
    exp_pw = pd.read_csv(os.path.join(filepath, 'transm.csv'), index_col=0)
    exp_cov.columns=exp_pw.E
    exp_cov.index=exp_pw.E
    exp_cov.index.name = None


    #%% Define experiment

    ac = 0.81271  # scattering radius in 1e-12 cm 
    M = 180.948030  # amu of target nucleus
    m = 1           # amu of incident neutron
    I = 3.5         # intrinsic spin, positive parity
    i = 0.5         # intrinsic spin, positive parity
    l_max = 1       # highest order l-wave to consider


    spin_groups = [ (3.0,1,0) ]
    average_parameters = pd.DataFrame({ 'dE'    :   {'3.0':8.79, '4.0':4.99},
                                        'Gg'    :   {'3.0':64.0, '4.0':64.0},
                                        'gn2'    :   {'3.0':46.4, '4.0':35.5}  })

    Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                    input_options={},
                                    spin_groups=spin_groups,
                                    average_parameters=average_parameters )   


    #%% initialize the data objects

    from ATARI.utils.misc import fine_egrid 
    from ATARI.utils.io.datacontainer import DataContainer
    from ATARI.utils.io.pointwise import PointwiseContainer
    from ATARI.utils.io.parameters import TheoreticalParameters, ExperimentalParameters

    threshold_0T = 1e-2
    exp_par = ExperimentalParameters(0.067166, 0, threshold_0T)
    theo_par = TheoreticalParameters(Ta_pair, theo_resladder, 'theo')

    pwfine = pd.DataFrame({'E':fine_egrid(exp_pw.E,100)})
    pw = PointwiseContainer(exp_pw, pwfine)
    pw.add_experimental(exp_pw, exp_cov, exp_par)
    pw.add_model(theo_par, exp_par)


    dc = DataContainer(pw, exp_par, theo_par,{})

    # %% Step 0, reduce initial feature bank for computational speed

    import classes as cls

    average_parameters.loc[:,['Gn']] = average_parameters['gn2']/12.5
    Elam_features, Gtot_features = fn.get_parameter_grid(pw.exp.E, average_parameters, '3.0', 1e0, 2e0)
    Gtot_features = np.append(Gtot_features, np.round(np.array(theo_resladder.Gt),1)*1e-3 )
    Elam_features = np.append(Elam_features, np.round(np.array(theo_resladder.E),1))
    # Elam_features = np.round(np.array(theo_resladder.E),1)
    # Gtot_features = np.array(theo_resladder.Gt)*1e-3
    # Elam_features = np.array(theo_resladder.E)

    w_threshold = 1e-6
    prob = cls.ProblemHandler(w_threshold)

    fb0 = prob.get_FeatureBank(dc, Elam_features, Gtot_features)
    inp0 = prob.get_MatrixInputs(dc, fb0)

    sol_lp0 = cls.Solvers.solve_linear_program(inp0)
    print(f'Features before: {fb0.nfeatures}')
    print(f'Features reduced with LP: {np.count_nonzero(sol_lp0>0)}')



    # %%  Step 1, solve unconstrained problem

    qpopt = cls.QPopt(verbose=True,
                    abstol = 1e-8,
                    reltol = 1e-8,
                    feastol=1e-7,
                    maxiters = 200)

    fb1, sol_lp0_ereduced = prob.reduce_FeatureBank(fb0, sol_lp0)
    inp1 = prob.get_MatrixInputs(dc, fb1)
    # fb1 = fb0
    # inp1 = inp0
    fb1.solution_ws = cls.Solvers.solve_quadratic_program(inp1, qpopt)

    # %% [markdown]
    # ## Step 2, run bisection routine
    def solve_qp_w_constraint(inp_uncon, wcon, qpopt: cls.QPopt):
        inp_con = prob.get_ConstrainedMatrixInputs(inp_uncon, wcon)
        sol = cls.Solvers.solve_quadratic_program(inp_con, qpopt)
        return sol


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
                        sol_ws[:,ires_sol] = con_sol_ws

                elif numres_sol > current_target_ires:
                    # record other if in target numres
                    if numres_sol in target_numres:
                        if wcon[ires_sol] < new_w:
                            wcon[ires_sol] = new_w
                            sol_ws[:,ires_sol] = con_sol_ws
                    # bisect again to find current target
                    wcon, sol_ws, save_all, searching = bisect_and_solve(current_target_ires, minval, new_w, target_numres, wcon, sol_ws, save_all, searching=searching)

                elif numres_sol < current_target_ires:
                    # record other if in target numres
                    if numres_sol in target_numres:
                        if wcon[ires_sol] < new_w:
                            wcon[ires_sol] = new_w
                            sol_ws[:,ires_sol] = con_sol_ws
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
            sol_ws = np.zeros((len(min_wcon_solw), len(target_numres)))
            # add min wconstraint and solution
            if min_wcon == 0:
                min_wcon = 1e-10
            target_wcon[0] = min_wcon
            sol_ws[:, 0] = min_wcon_solw
            # add max if max numres (unconstrained solve) is in the target
            if max_numres in target_numres:
                target_wcon[-1] = max_wcon
                sol_ws[:, -1] = max_solw
        
        elif target_wcon is None or sol_ws is None:
            raise ValueError("Either target_wcon or sol_ws is None while the other is populated, check input.")
        else:
            elements_to_add = len(target_numres)-len(target_wcon)
            assert elements_to_add >= 0, "target_numres is shorter than target_wcon"
            target_wcon = np.append(target_wcon, [0]*elements_to_add)
            sol_ws = np.append(sol_ws, np.zeros((np.shape(sol_ws)[0], elements_to_add)), axis=1)
        return target_wcon, sol_ws

    # %%

    # determine mins and maxes
    # qpopt.verbose=True; qpopt.maxiters=200
    min_wcon = prob.get_MinSolvableWeight(fb1.nfeatures, inp1)
    max_wcon = np.sum(fb1.solution_ws)
    max_numres = np.count_nonzero(fb1.solution_ws>prob.w_threshold)
    min_wcon_solw = solve_qp_w_constraint(inp1, min_wcon*1.01, qpopt)
    min_numres = np.count_nonzero(min_wcon_solw>prob.w_threshold)


    # solve_qp_w_constraint(inp1, min_wcon*1.001, qpopt)
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
        target_wcon, target_sol_ws, save_all, _ = bisect_and_solve(current_target_ires,
                                                                    minwcon, maxwcon,
                                                                    target_numres, target_wcon, target_sol_ws, save_all)

    # print(target_numres)
    # print(target_wcon)

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


    # %% [markdown]

    # ### to just combine if E are the exact same
    def calculate_combined_parameters(x):
        if len(x) == 1:
            return x #[['Gt', 'Gnx', 'Gg', 'w']]
        else:
            # Gnx = (x['w']*x['Gt']).sum()
            # Gt = (x['w']*x['Gt']**2).sum() / Gnx
            # w = Gnx/Gt
            # Gg = Gt-Gnx
            w = x['w'].sum()
            Gnx = w*1e3
            Gg = np.average(x['Gg'], weights=x['w'])
            Gt = Gnx+Gg
        return pd.DataFrame({'E':x['E'].unique(), 'Gt':Gt, 'Gnx':Gnx, 'Gg':Gg, 'w':w}, index=[0]) # pd.DataFrame({'Gt':Gt, 'Gnx':Gnx, 'Gg':Gg, 'w':w}, index=[0])

    print('Combining features at the same energy location')

    ### Here's where I handle resonances at the same energies and those with weights very small
    integer_resonance_solutions = {}
    for numres in target_numres[target_wcon!=0]:
        ires_featurebank = integer_feature_solutions[numres]
        ires_resladder = ires_featurebank.get_parameter_solution()

        ires_resladder_combined = ires_resladder.groupby(ires_resladder['E'], group_keys=False).apply(calculate_combined_parameters).reset_index(drop=True)
        ires_resladder_combined = ires_resladder_combined.drop(ires_resladder_combined[ires_resladder_combined['w'] < prob.w_threshold].index)
        new_numres = len(ires_resladder_combined)

        ires_resladder_combined = fill_resonance_ladder(ires_resladder_combined, Ta_pair, J=3.0, chs=1, lwave=0.0, J_ID=1)
        integer_resonance_solutions[new_numres] = {'prior':ires_resladder_combined}

        # add prior to dc
        est_par = TheoreticalParameters(Ta_pair, ires_resladder_combined, label=f'{new_numres}_prior')
        dc.add_estimate(est_par)

    print(f'Surviving integer number of resonance solutions: {list(integer_resonance_solutions.keys())}')

    # %% [markdown] Step 4, run GLLS on transmission with reduced, unconstrained solution from 3 as prior


    from ATARI.sammy_interface import sammy_functions, sammy_classes
    sammyRTO = sammy_classes.SammyRunTimeOptions(
        path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
        model = 'SLBW',
        reaction = 'transmission',
        solve_bayes = True,
        experimental_corrections = 'no_exp',
        one_spingroup = False,
        energy_window = None,
        sammy_runDIR = 'SAMMY_runDIR',
        keep_runDIR = False,
        shell = 'zsh'
        )

    print(f'Now running SAMMY Bayes for each integer resonance solution')

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
        )

        lst, posterior = sammy_functions.run_sammy(sammyINP, sammyRTO)
        ## if posterior is worse than prior, re-run
        posterior = fill_resonance_ladder(posterior, Ta_pair, J=3.0, chs=1, lwave=0.0, J_ID=1)
        integer_resonance_solutions[numres]['posterior'] = posterior

        est_par = TheoreticalParameters(Ta_pair, posterior, label=f'{numres}_post')
        dc.add_estimate(est_par)

    # %% LRT

    ### Calculate Chi2 on trans
    # [ (chi2_val(dc.pw.exp[f'{numres}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT), numres) for numres in dc.est_par.keys()]

    print('Now running recursive likelihood ratio test')
    posterior_ires_chi2 = [ (int(key.split('_')[0]),
                            chi2_val(dc.pw.exp[f'{key}_trans'], dc.pw.exp.exp_trans, dc.pw.CovT))
                                for key in dc.est_par.keys() if key.split('_')[1]=='post']
    posterior_ires_chi2 = np.array(posterior_ires_chi2)
    posterior_ires_chi2 = posterior_ires_chi2[posterior_ires_chi2[:, 0].argsort()]

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
        likelihood = likelihood_val( np.array(dc.pw.exp[f'{int(posterior_ires_chi2[i][0])}_post_trans']) )

        print(np.log(likelihood))
        if np.log(likelihood) >= -100:
            istart = i
            break


    inull = istart 
    ialt = inull
    iend = np.shape(posterior_ires_chi2)[0]

    significance_level = 0.05

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

    final_estimate = dc.est_par[f'{selected_model_ires}_post'].resonance_ladder
    final_par = TheoreticalParameters(Ta_pair, final_estimate, 'final')
    dc.add_estimate(final_par)
    est_chi_square = (dc.pw.exp.exp_trans-dc.pw.exp.final_trans) @ inv(dc.pw.CovT) @ (dc.pw.exp.exp_trans-dc.pw.exp.final_trans).T
    sol_chi_square = (dc.pw.exp.exp_trans-dc.pw.exp.theo_trans) @ inv(dc.pw.CovT) @ (dc.pw.exp.exp_trans-dc.pw.exp.theo_trans).T
    from scipy import integrate
    est_sol_MSE = integrate.trapezoid((dc.pw.fine.theo_xs-dc.pw.fine.final_xs)**2, dc.pw.fine.E)

    return est_sol_MSE #final_estimate
    



#%% Command line interface


import sys
args = 29 # sys.argv[1]
import time

start_time = time.time()
final_estimate = main(args)
end_time = time.time()
elapsed_time = end_time - start_time


# final_estimate['tfit'] = np.ones(len(final_estimate))*elapsed_time
# final_estimate.to_csv(f'./par_est_{casenum}.csv')
print()
print(f'Case: {args}')
print(f'Final MSE: {final_estimate}')
print(f'Time: {elapsed_time}')
print()