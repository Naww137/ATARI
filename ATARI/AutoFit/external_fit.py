from ATARI.sammy_interface.sammy_io import *
from ATARI.sammy_interface.sammy_functions import *
from ATARI.sammy_interface.sammy_deriv import get_derivatives
from ATARI.sammy_interface.convert_u_p_params import p2u_E, p2u_g, p2u_n,   u2p_E, u2p_g, u2p_n,   du_dp_E, du_dp_g, du_dp_n
import numpy as np
import pandas as pd
from ATARI.sammy_interface import sammy_classes


def get_Gs_Ts(resonance_ladder, 
              particle_pair,
              experiments,
              datasets, 
              covariance_data,
              rto):
    
    Gs = []; Ts = []; pw_list = []
    for exp, dat, exp_cov in zip(experiments, datasets, covariance_data):
        sammyINP = sammy_classes.SammyInputData(particle_pair, resonance_ladder, experiment=exp, experimental_data = dat, experimental_covariance=exp_cov)
        sammy_out = get_derivatives(sammyINP, rto, get_theo=True, u_or_p='u')
        Gs.append(sammy_out.derivatives)
        
        if exp.reaction == "transmission":
            key = "theo_trans"
        else:
            key = "theo_xs"
        Ts.append(sammy_out.pw[key].values)
        pw_list.append(sammy_out.pw)

    return Gs, Ts, pw_list

from ATARI.utils.stats import add_normalization_uncertainty_to_covariance
def get_Ds_Vs(experiments, datasets, covariance_data, normalization_uncertainty = 0.038):
    Ds = []; covs = []
    for exp, dat, exp_cov in zip(experiments, datasets, covariance_data):
        # sort everything first
        d = dat.sort_values("E")
        d.reset_index(drop=True, inplace=True)

        Ds.append(d["exp"].values)
        if not exp_cov:
            c = np.diag(d["exp_unc"].values**2)
            c = add_normalization_uncertainty_to_covariance(c, d["exp"].values, normalization_uncertainty)
            covs.append(c)
        else:
            diag_stat = exp_cov["diag_stat"].sort_values("E")
            diag_stat.reset_index(drop=True)
            Jac_sys = exp_cov["Jac_sys"].sort_index(axis=1)
            min_max_E = (np.min(d.E), np.max(d.E))
            mask = (diag_stat.index>=min_max_E[0]) & (diag_stat.index<=min_max_E[1])
            J = Jac_sys.values[:, mask]
            covs.append(np.diag(diag_stat[mask]['var_stat'].values) + J.T @ exp_cov['Cov_sys'] @ J)
    return Ds, covs

def zero_G_at_no_vary(G, par):
    # Need to remove all zero cols so inv(hessian) is not singular
    # nonzero_columns = np.any(G != 0, axis=0)
    # used_cols = np.nonzero(nonzero_columns)[0]
    # G = G[:,used_cols]
    # Pp = P
    # Pp[used_cols] += (Mp @ G.T @ VDT)

    indices_e = 3*par.index[par['varyE'] == 0].to_numpy()
    indices_g = 3*par.index[par['varyGg'] == 0].to_numpy() + 1
    indices_n = 3*par.index[par['varyGn1'] == 0].to_numpy()  + 2
    G[:,indices_e] = 0.0
    G[:,indices_g] = 0.0
    G[:,indices_n] = 0.0
    return G


def get_Pu_vec_and_indices(resonance_ladder,
                           particle_pair,
                           external_resonance_indices=[]
                           ):

    E  = resonance_ladder['E'].to_numpy()
    Gg = resonance_ladder['Gg'].to_numpy()
    Gn = resonance_ladder['Gn1'].to_numpy()
    J_ID = resonance_ladder['J_ID'].to_numpy(dtype=int)

    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]

    ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((ue, ug, un)).reshape(-1,)

    ires = np.arange(0,len(resonance_ladder), 1)
    iE = 3*ires
    igg = 3*ires + 1
    ign = 3*ires + 2

    iext = []
    for i in external_resonance_indices:
        iext.extend([3*i, 3*i+1, 3*i+2])
    iext = np.array(iext)

    iE_no_step = 3*resonance_ladder.index[resonance_ladder['varyE'] == 0].to_numpy()
    igg_no_step = 3*resonance_ladder.index[resonance_ladder['varyGg'] == 0].to_numpy() + 1
    ign_no_step = 3*resonance_ladder.index[resonance_ladder['varyGn1'] == 0].to_numpy()  + 2
    i_no_step = np.concatenate([iE_no_step, igg_no_step, ign_no_step])

    return Pu, iE, igg, ign, iext, i_no_step

def get_p_resonance_ladder_from_Pu_vector(Pu, 
                                          initial_reslad,
                                          particle_pair
                                          ):
    J_ID = initial_reslad['J_ID'].to_numpy(dtype=int)
    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]

    Pu = Pu.reshape(-1,3)
    Eu  = Pu[:,0]
    Ggu = Pu[:,1]
    Gnu = Pu[:,2]
    Pp = np.zeros_like(Pu)
    Pp[:,0] = u2p_E(Eu)
    Pp[:,1] = u2p_g(Ggu)
    Pp[:,2] = u2p_n(Gnu, Pp[:,0], L[J_ID-1], particle_pair)

    par_post = pd.DataFrame(Pp, columns=['E', 'Gg', 'Gn1'])
    par_post['J_ID']    = initial_reslad['J_ID']
    par_post['varyE']   = initial_reslad['varyE']
    par_post['varyGn1'] = initial_reslad['varyGn1']
    par_post['varyGg']  = initial_reslad['varyGg']

    return par_post

def get_derivatives_for_step(rto, D, V, 
                             datasets, covariance_data, ### dont need these two things if I update get_derivatives function Cole created
                             res_lad, 
                             particle_pair,
                             experiments, 
                             zero_derivs_at_no_vary = True
                             ):
    Gs, Ts, sammy_pws = get_Gs_Ts(res_lad, particle_pair, experiments, datasets, covariance_data, rto)
    G = np.concatenate(Gs, axis=0)
    T = np.concatenate(Ts)

    # zero derivative for parameters not varied
    if zero_derivs_at_no_vary:
        G = zero_G_at_no_vary(G, res_lad)

    Vinv = np.linalg.inv(V)

    # calculate derivative and chi2
    dchi2_dpar = - 2 * G.T @ Vinv @ (D - T)
    hessian_approx = G.T @ Vinv @ G
    chi2 = (D-T).T @ Vinv @ (D-T)

    return chi2, dchi2_dpar, hessian_approx, sammy_pws


def get_derivatives_for_step_batched(rto, D, V, 
                             datasets, covariance_data, ### dont need these two things if I update get_derivatives function Cole created
                             res_lad, 
                             particle_pair,
                             experiments, 
                             zero_derivs_at_no_vary = True,
                             batches = 25,
                             batch_components = False
                             ):
    Gs, Ts, sammy_pws = get_Gs_Ts(res_lad, particle_pair, experiments, datasets, covariance_data, rto)
    G = np.concatenate(Gs, axis=0)
    T = np.concatenate(Ts)

    # zero derivative for parameters not varied
    if zero_derivs_at_no_vary:
        G = zero_G_at_no_vary(G, res_lad)

    chi2 = (D-T).T @ np.linalg.inv(V) @ (D-T)

    N = G.shape[0]
    remainder = N%batches
    N_per_batch = int((N-remainder)/batches)
    rand_int = np.random.choice(N, size=N, replace=False) 
    if batch_components:
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
    dchi2_dpar_batch = []; hessian_approx_batch = []; chi2_batch = []
    for i in range(batches):
        if i == batches - 1:
            end = (i+1)*N_per_batch + remainder
        else:
            end = (i+1)*N_per_batch
        index = rand_int[i*N_per_batch:end]
        if batch_components:
                s_i = s[index]
                U_i = U[:, index]
                Vt_i = Vt[index, :]
                Vinv = Vt_i.T @ np.linalg.inv(np.diag(s_i)) @ U_i.T
                dchi2_dpar_batch.append( - 2 * G.T @ Vinv @ (D - T))
                hessian_approx_batch.append( G.T @ Vinv @ G)
                # chi2_batch.append((D-T).T @ np.linalg.inv(V) @ (D-T))
        else:
            print("WARNING: Check this implementation")
            # calculate derivative and chi2 for batch
            dchi2_dpar_batch.append( - 2 * G[index,:].T @ np.linalg.inv(V[index,:][:,index]) @ (D[index] - T[index]))
            hessian_approx_batch.append( G[index,:].T @ np.linalg.inv(V[index,:][:,index]) @ G[index,:])
            # chi2_batch.append((D-T).T @ np.linalg.inv(V) @ (D-T))

    return chi2, dchi2_dpar_batch, hessian_approx_batch, chi2_batch, sammy_pws


def evaluate_chi2_location_and_gradient(rto, Pu, starting_ladder, particle_pair, D, V, datasets, covariance_data, experiments, batches=1, batch_components = False):
    res_lad = get_p_resonance_ladder_from_Pu_vector(Pu, starting_ladder,particle_pair)

    if batches == 1:
        chi2, dchi2_dpar, hessian_approx, sammy_pws = get_derivatives_for_step(rto, D, V, datasets, covariance_data, res_lad, particle_pair,experiments)
    else:
        chi2, dchi2_dpar, hessian_approx, chi2_batch, sammy_pws = get_derivatives_for_step_batched(rto, D, V, datasets, covariance_data, res_lad, particle_pair,experiments, batches=batches, batch_components=batch_components)
    # dchi2_dpar = np.clip(dchi2_dpar, -100, 100) ### Gradient cliping (value, could also do norm) for exploding gradients and numerical stability
    return chi2, dchi2_dpar, hessian_approx, sammy_pws, res_lad



def get_regularization_location_and_gradient(Pu, ign, iE, iext,
        lasso=False, lasso_parameters =     {"lambda":1,    "gamma":2},
        ridge = False, ridge_parameters =   {"lambda":1,    "gamma":2},
        elastic_net=True,elastic_net_parameters = {"lambda":1, "gamma":2, "alpha":0.5} ):
    
    if lasso:
        ign_internal = [i for i in ign if i not in iext]
        gn = Pu[ign_internal]
        lasso_pen = lasso_parameters["lambda"]* np.sum( gn*np.sign(gn) )
        lasso_gradient = np.zeros_like(Pu)
        lasso_gradient[ign_internal] = lasso_parameters["lambda"]*np.sign(gn)
        if lasso_parameters["weights"] is not None:
            w = lasso_parameters["weights"]
            lasso_pen = lasso_parameters["lambda"]* np.sum( gn*np.sign(gn) / (w[ign_internal]*np.sign(w[ign_internal]))**lasso_parameters["gamma"])
            lasso_gradient[ign_internal] /= (w[ign_internal]*np.sign(w[ign_internal]))**lasso_parameters["gamma"]
        pen = lasso_pen
        grad = lasso_gradient
        assert(not ridge); assert(not elastic_net)

    elif ridge:
        ign_internal = [i for i in ign if i not in iext]
        gn = Pu[ign_internal]
        ridge_pen = ridge_parameters["lambda"]* np.sum( gn**2 )
        ridge_gradient = np.zeros_like(Pu)
        ridge_gradient[ign_internal] = ridge_parameters["lambda"]*2*gn
        if ridge_parameters["weights"] is not None:
            raise ValueError("Weighted ridge not yet implemented")
        pen = ridge_pen
        grad = ridge_gradient
        assert(not lasso); assert(not elastic_net)

    elif elastic_net:
        ign_internal = [i for i in ign if i not in iext]
        gn = Pu[ign_internal]
        elastic_net_pen = elastic_net_parameters["lambda"]* np.sum(elastic_net_parameters["alpha"]*gn**2 + (1-elastic_net_parameters["alpha"])*gn*np.sign(gn))
        elastic_net_gradient = np.zeros_like(Pu)
        elastic_net_gradient[ign_internal] = elastic_net_parameters["lambda"]*(elastic_net_parameters["alpha"]*2*gn + 1 - elastic_net_parameters["alpha"])
        if elastic_net_parameters["weights"] is not None:
            raise ValueError("Weighted elastic net not yet implemented")
        pen = elastic_net_pen
        grad = elastic_net_gradient
        assert(not ridge); assert(not lasso)

    else:
        pen = 0
        grad = np.zeros_like(Pu)

    return pen, grad





def take_step(Pu, alpha, dchi2_dpar, hessian_approx, dreg_dpar, iE, ign, gaus_newton=False, momentum = 0):
    
    alpha_vec = np.ones_like(Pu)*alpha
    alpha_vec[iE] = alpha/100 #* Pu[ign] #alpha_vec[iE]
    # alpha_vec[ign] = alpha * Pu[ign]
    if gaus_newton: 
        dampening = 1/alpha_vec
        chi2_step = np.linalg.inv(dampening*np.diag(np.ones_like(Pu)) + hessian_approx) @ dchi2_dpar
    else:
        chi2_step = alpha_vec*dchi2_dpar
    
    total_gradient = chi2_step + alpha*dreg_dpar + momentum
    total_gradient[iE] = np.where(total_gradient[iE]>0.1, 0.1, total_gradient[iE])
    total_gradient[iE] = np.where(total_gradient[iE]<-0.1, -0.1, total_gradient[iE])

    return Pu - total_gradient



def fit(rto, 
        starting_ladder, 
        external_resonance_indices, 
        particle_pair,
        D, V, experiments, datasets, covariance_data,

        steps = 100, 
        thresh = 0.01, 
        alpha = 1e-6, 
        print_bool = True, 
        
        gaus_newton = False, 

        LevMar = True, 
        LevMarV = 2, 
        LevMarVd = 5, 
        maxV = 1e-4, 
        minV=1e-8, 
        
        lasso = False,
        lasso_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        ridge = False,
        ridge_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        elastic_net = False,
        elastic_net_parameters = {"lambda":1, 
                                "gamma":0,
                                "alpha":0.7}
        ):

    # alpha = np.array(alpha)
    # if np.any(alpha <= 0):
    #     raise ValueError("learn_rate 'alpha' must be greater than zero")

    saved_res_lads = []
    save_Pu = []
    saved_pw_lists = []
    saved_gradients = []
    chi2_log = []
    obj_log = []

    Pu_next, iE, igg, ign, iext, i_no_step = get_Pu_vec_and_indices(starting_ladder, particle_pair, external_resonance_indices)
    for istep in range(steps):

        # Get current location derivatives and objective function values
        chi2, dchi2_dpar, hessian_approx, sammy_pws, res_lad = evaluate_chi2_location_and_gradient(rto, Pu_next, starting_ladder, particle_pair,D, V, datasets,covariance_data,experiments)
        reg_pen, dreg_dpar = get_regularization_location_and_gradient(Pu_next, ign, iE, iext,
                                                                    lasso =lasso, lasso_parameters = lasso_parameters,
                                                                    ridge = ridge, ridge_parameters = ridge_parameters,
                                                                    elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)
        obj = chi2 + reg_pen
        
        if istep > 1:
            
            if LevMar:
                assert(LevMarV>1)
                if obj < obj_log[istep-1]:
                    alpha *= LevMarV
                    alpha = min(alpha,maxV)
                else:
                    if print_bool:
                        print(f"Decrease alpha and repeat step {int(istep)}")
                        print(f"\t\t{np.round(float(alpha),8):<10}: {obj:.2f}\t{chi2:.2f}")
                    while True:  
                        alpha /= LevMarVd
                        alpha = max(alpha, minV)
                        Pu_temp = take_step(Pu, alpha, dchi2_dpar, hessian_approx, dreg_dpar, iE, ign, gaus_newton=gaus_newton)
                        chi2_temp, dchi2_dpar_temp, hessian_approx_temp, sammy_pws_temp, res_lad_temp = evaluate_chi2_location_and_gradient(rto, Pu_temp, starting_ladder, particle_pair,D, V, datasets,covariance_data,experiments)
                        reg_pen_temp, dreg_dpar_temp = get_regularization_location_and_gradient(Pu_temp, ign, iE, iext,
                                                                                                lasso =lasso, lasso_parameters = lasso_parameters,
                                                                                                ridge = ridge, ridge_parameters = ridge_parameters,
                                                                                                elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)
                        obj_temp = chi2_temp + reg_pen_temp

                        if print_bool:
                            print(f"\t\t{np.round(float(alpha),8):<10}: {obj_temp:.2f}\t{chi2_temp:.2f}")
                        if obj_temp < obj_log[istep-1] or alpha==minV:
                            obj, chi2, dchi2_dpar, hessian_approx, dreg_dpar, sammy_pws, res_lad = obj_temp, chi2_temp, dchi2_dpar_temp, hessian_approx_temp, dreg_dpar_temp, sammy_pws_temp, res_lad_temp
                            Pu_next = Pu_temp
                            break
                        else:
                            pass
            
            ### convergence check
            Dobj = obj_log[istep-1] - obj
            if Dobj < thresh:
                if Dobj < 0:
                    criteria = f"Obj increased, taking solution {istep-1}"
                    if LevMar and alpha==minV:
                        criteria = f"Step size below minimum value, taking solution {istep-1}"
                    if print_bool:
                        print(criteria)
                    print(max(istep-1, 0))
                    break
                else:
                    criteria = "Obj improvement below threshold"
                if print_bool:
                    print(criteria)
                print(istep)
                break
        
        if print_bool:
            print(f"{int(istep)}\t{np.round(alpha,8):<10}:\t{obj:.2f}\t{chi2:.2f}")

        ### update Pu to Pu_next and save things
        Pu = Pu_next
        obj_log.append(obj)
        chi2_log.append(chi2)
        gradient = dchi2_dpar
        saved_gradients.append(gradient)
        save_Pu.append(Pu)
        saved_pw_lists.append(sammy_pws)
        saved_res_lads.append(copy(res_lad))

        ### step coeficients
        Pu_next = take_step(Pu, alpha, dchi2_dpar, hessian_approx, dreg_dpar, iE, ign, gaus_newton=gaus_newton)

    return saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log



def fit_and_characterize(rto, 
                        starting_ladder, 
                        external_resonance_indices, 
                        particle_pair,
                        D, V, experiments, datasets, covariance_data,
                        steps, thresh, alpha, print_bool, gaus_newton, 
                        LevMar,LevMarV,LevMarVd,maxV,minV,
                        lasso,lasso_parameters,
                        ridge, ridge_parameters,
                        elastic_net, elastic_net_parameters):
    
    saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log = fit(rto, 
                                                                                    starting_ladder, 
                                                                                    external_resonance_indices, 
                                                                                    particle_pair,
                                                                                    D, V, experiments, datasets, covariance_data,
                                                                                    steps, thresh, alpha, print_bool, gaus_newton, 
                                                                                    LevMar,LevMarV,LevMarVd,maxV,minV,
                                                                                    lasso,lasso_parameters,
                                                                                    ridge, ridge_parameters,
                                                                                    elastic_net, elastic_net_parameters)
    
    rto_temp = deepcopy(rto)
    rto_temp.bayes = False
    inpyw = sammy_classes.SammyInputDataYW(
                particle_pair = particle_pair,
                resonance_ladder = saved_res_lads[-1],  

                datasets= datasets,
                experiments = experiments,
                experimental_covariance= covariance_data)

    sammyOUT = run_sammy_YW(inpyw, rto_temp)

    return sammyOUT










def sgd(rto, 
        starting_ladder, 
        external_resonance_indices, 
        particle_pair,
        D, V, experiments, datasets, covariance_data,

        steps = 100, 
        thresh = 0.01,

        alpha = 1e-3, 
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-8,

        print_bool = True, 

        batches = 25, 
        batch_components = False,

        gaus_newton = False, 
        
        lasso = False,
        lasso_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        ridge = False,
        ridge_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        elastic_net = False,
        elastic_net_parameters = {"lambda":1, 
                                "gamma":0,
                                "alpha":0.7},

        ):

    saved_res_lads = []
    save_Pu = []
    saved_pw_lists = []
    saved_gradients = []
    chi2_log = []
    obj_log = []

    Pu_next, iE, igg, ign, iext, i_no_step = get_Pu_vec_and_indices(starting_ladder, particle_pair, external_resonance_indices)
    

    ### Initialize adam parameters
    t = 0
    mt = np.zeros_like(Pu_next)
    vt = np.zeros_like(Pu_next)
    # decay_rate = np.array(decay_rate)
    # if np.any(decay_rate < 0) or np.any(decay_rate > 1):
    #     raise ValueError("'decay_rate' must be between zero and one")
    # alpha = np.array(alpha)
    # if np.any(alpha <= 0):
    #     raise ValueError("learn_rate 'alpha' must be greater than zero")

    N = V.shape[0]
    remainder = N%batches
    N_per_batch = int((N-remainder)/batches)
    U, s, Vt = np.linalg.svd(V, full_matrices=False)


    for istep in range(steps):
        
        # Get current location derivatives and objective function values, save things 
        # chi2, dchi2_dpar, hessian_approx, sammy_pws, res_lad = evaluate_chi2_location_and_gradient(rto, Pu_next, starting_ladder, particle_pair,D, V, datasets,covariance_data,experiments, batches=batches, batch_components=batch_components)

        #### Need to split regularization gradient into batches too
        dreg_dpar = np.zeros_like(Pu_next)
        reg_pen = 0 ### for now no regularization
        # reg_pen_temp, dreg_dpar_temp = get_regularization_location_and_gradient(Pu_next, ign, iE, iext,
        #                                                                         lasso =lasso, lasso_parameters = lasso_parameters,
        #                                                                         ridge = ridge, ridge_parameters = ridge_parameters,
        #                                                                         elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)

    
        rand_int = np.random.choice(N, size=N, replace=False) 
        # Performing minibatch moves
        for i in range(batches):
            Pu = Pu_next
            t += 1
            # Pu_next = take_step(Pu, alpha, dchi2_dpar[ibatch], hessian_approx[ibatch], dreg_dpar[ibatch], iE, gaus_newton=gaus_newton, momentum=momentum)

            # if i == batches - 1:
            #     end = (i+1)*N_per_batch + remainder
            # else:
            #     end = (i+1)*N_per_batch
            # index = rand_int[i*N_per_batch:end]
            index = np.random.choice(N, size=int(N/batches), replace=False) 
            s_i = s[index]
            U_i = U[:, index]
            Vt_i = Vt[index, :]

            res_lad = get_p_resonance_ladder_from_Pu_vector(Pu, starting_ladder, particle_pair)
            Gs, Ts, sammy_pws = get_Gs_Ts(res_lad, particle_pair, experiments, datasets, covariance_data, rto)
            G = np.concatenate(Gs, axis=0); T = np.concatenate(Ts)

            chi2 = (D-T).T @ np.linalg.inv(V) @ (D-T)
            obj = chi2 + reg_pen
            if print_bool:
                print(f"{int(t)}\t:\t{obj:.2f}\t{chi2:.2f}")
                
            if True: #zero_derivs_at_no_vary:
                G = zero_G_at_no_vary(G, res_lad)

            Vinv = Vt_i.T @ np.linalg.inv(np.diag(s_i)) @ U_i.T
            dchi2_dpar =  - 2 * G.T @ Vinv @ (D - T)
            hessian_approx =  G.T @ Vinv @ G
            #### The following should be more efficient but verify!
            # dchi2_dpar =  - 2 * G.T@Vt_i.T @ np.linalg.inv(np.diag(s_i)) @ U.T@(D - T)
            # hessian_approx =  G.T@Vt_i.T @ np.linalg.inv(np.diag(s_i)) @ U.T@ G

            # dchi2_dpar = np.clip(dchi2_dpar, -100, 100) ### Gradient cliping (value, could also do norm) for exploding gradients and numerical stability
            dObj_dtheta = dchi2_dpar + dreg_dpar
            mt = (beta_1 * mt + (1 - beta_1) * dObj_dtheta)
            vt = (beta_2 * vt + (1 - beta_2) * dObj_dtheta**2) 
            mt_hat = (mt / (1 - beta_1**t))
            vt_hat = (vt/ (1 - beta_2**t))
            
            Pu_next = (Pu - (alpha * mt_hat) / (np.sqrt(vt_hat) + epsilon))

            # obj_log.append(obj)
            chi2_log.append(chi2)
            saved_gradients.append(dchi2_dpar)
            save_Pu.append(Pu)
            saved_pw_lists.append(sammy_pws)
            saved_res_lads.append(copy(res_lad))
                
        print(f"Epoch {istep} done")
        

    return saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log