from copy import copy
import numpy as np
from ATARI.AutoFit.sammy_interface_bindings import Solver
from pandas import DataFrame

def bisect(x0,x1):
    return (x0+x1)/2

def get_starting_ladder_w0_l0(ridge_out, g0, l0_chi2frac=0.1):
    starting_ladder = copy(ridge_out.par_post)
    w0 = ridge_out.Pu_post
    w0_gn = ridge_out.Pu_post.reshape(-1,3)[:,2]
    initial_lasso_pen = np.sum(1/w0_gn**g0 *abs(w0_gn))
    initial_chi2 = np.sum(ridge_out.chi2_post)
    l0 = initial_chi2*l0_chi2frac/initial_lasso_pen
    # lmax = initial_chi2*1.5/initial_lasso_pen
    return starting_ladder, w0, l0

def get_new_lam(save_lambdas, save_nres, current_target_ires, lam, l0):
    temp = np.vstack([save_lambdas,save_nres])
    temp = temp[:,np.lexsort([temp[1,:],-temp[0,:]])]
    index = np.searchsorted(temp[1,:], current_target_ires, side='left')
    if index == 0:
        lam = lam + l0
        print(f"\nAdding to lambda: {lam}\n------\n")
    elif index == len(temp[1,:]):
        lam = lam - l0
    else:
        lam_max, lam_min = temp[0,index], temp[0,index-1]
        lam = bisect(lam_min, lam_max)
        print(f"\nBisecting lambda: {lam}\n------\n")
    return max(lam,0)
        


        
def get_target_ires(
                    target_ires:                int, 
                    solver:                     Solver, 
                    starting_ladder:            DataFrame,
                    external_resonance_indices: list, 
                    lasso_weights:              np.ndarray, 
                    lambda_step:                float, 
                    Gn_threshold:               float, 
                    resolve:                    bool = True,
                    ):
    
    lam = 0
    reslad = starting_ladder
    save_lambdas = []
    save_nres = []
    save_outs = []
    while True:

        solver.sammyINP.lasso_parameters["lambda"] = lam #, "gamma":lasso_gamma, 
        solver.sammyINP.lasso_parameters["weights"] = lasso_weights
        if resolve:
            lasso_out = solver.fit(starting_ladder, external_resonance_indices)
        else:
            lasso_out = solver.fit(reslad, external_resonance_indices)

        reslad = lasso_out.par_post
        numres = np.count_nonzero(reslad.Gn1>Gn_threshold)-len(external_resonance_indices)
        print(f"\n------\nNres: {numres}, lambda: {lam}")
        save_lambdas.append(lam); save_nres.append(numres); save_outs.append(lasso_out)
    
        if numres == target_ires:
            break
        else:
            lam = get_new_lam(save_lambdas, save_nres, target_ires, lam, lambda_step)
            if np.any(np.isclose(save_lambdas, lam, atol=1e-2)):
                break
    
    if save_nres[-1] == target_ires:
        print(f"\n==========\nSuccessfully found model with {target_ires} non-zero resonances")
        return save_outs[-1], save_outs, save_nres, save_lambdas
    else:
        print(f"\n==========\nCould NOT find model with {target_ires} non-zero resonances")
        return None, save_outs, save_nres, save_lambdas

