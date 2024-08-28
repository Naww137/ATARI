from ATARI.sammy_interface.sammy_io import *
from ATARI.sammy_interface.sammy_functions import *
from ATARI.sammy_interface.sammy_deriv import get_derivatives
from ATARI.utils.stats import add_normalization_uncertainty_to_covariance
from ATARI.sammy_interface.convert_u_p_params import p2u_E, p2u_g, p2u_n,   u2p_E, u2p_g, u2p_n,   get_Pu_vec_from_reslad
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

def get_Ds_Vs(datasets, covariance_data, normalization_uncertainty = 0.038):
    Ds = []; covs = []
    for dat, exp_cov in zip(datasets, covariance_data):
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

    # E  = resonance_ladder['E'].to_numpy()
    # Gg = resonance_ladder['Gg'].to_numpy()
    # Gn = resonance_ladder['Gn1'].to_numpy()
    # J_ID = resonance_ladder['J_ID'].to_numpy(dtype=int)

    # L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    # for jpi, mean_params in particle_pair.spin_groups.items():
    #     jid = mean_params['J_ID']
    #     L[jid-1] = mean_params['Ls'][0]

    # ue = p2u_E(E)
    # ug = p2u_g(Gg)
    # un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    # Pu = np.column_stack((ue, ug, un)).reshape(-1,)
    Pu = get_Pu_vec_from_reslad(resonance_ladder, particle_pair)

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
                             zero_derivs_at_no_vary = True,
                             V_is_inv=False
                             ):
    Gs, Ts, sammy_pws = get_Gs_Ts(res_lad, particle_pair, experiments, datasets, covariance_data, rto)
    G = np.concatenate(Gs, axis=0)
    T = np.concatenate(Ts)

    # zero derivative for parameters not varied
    if zero_derivs_at_no_vary:
        G = zero_G_at_no_vary(G, res_lad)

    if V_is_inv:
        Vinv = V
    else:
        Vinv = np.linalg.inv(V)

    # calculate derivative and chi2
    dchi2_dpar = - 2 * G.T @ Vinv @ (D - T)
    hessian_approx = G.T @ Vinv @ G
    chi2 = (D-T).T @ Vinv @ (D-T)

    return chi2, dchi2_dpar, hessian_approx, sammy_pws


def evaluate_chi2_location_and_gradient(rto, Pu, starting_ladder, particle_pair, D, V, datasets, covariance_data, experiments, V_is_inv=False):
    res_lad = get_p_resonance_ladder_from_Pu_vector(Pu, starting_ladder,particle_pair)
    chi2, dchi2_dpar, hessian_approx, sammy_pws = get_derivatives_for_step(rto, D, V, datasets, covariance_data, res_lad, particle_pair,experiments, V_is_inv=V_is_inv)
    # dchi2_dpar = np.clip(dchi2_dpar, -100, 100) ### Gradient cliping (value, could also do norm) for exploding gradients and numerical stability
    return chi2, dchi2_dpar, hessian_approx, sammy_pws, res_lad



def get_regularization_location_and_gradient(Pu, ign, iE, iext,
        lasso=False, lasso_parameters =     {"lambda":1,    "gamma":2},
        ridge = False, ridge_parameters =   {"lambda":1,    "gamma":2},
        elastic_net=False,elastic_net_parameters = {"lambda":1, "gamma":2, "alpha":0.5} ):
    
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
    # alpha_vec[ign] = alpha * Pu[ign]**2
    if gaus_newton: 
        dampening = 1/alpha_vec
        chi2_step = np.linalg.inv(dampening*np.diag(np.ones_like(Pu)) + hessian_approx) @ dchi2_dpar
    else:
        chi2_step = alpha_vec*dchi2_dpar
    
    total_gradient = chi2_step + alpha*dreg_dpar + momentum
    max_estep = 0.05 #Pu[ign]**2/10
    total_gradient[iE] = np.where(abs(total_gradient[iE])>max_estep, np.sign(total_gradient[iE])*max_estep, total_gradient[iE])
    # total_gradient[iE] = np.where(total_gradient[iE]<-0.05, -0.1, total_gradient[iE])

    return Pu - total_gradient



from ATARI.syndat.control import syndatOPT
from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.AutoFit.external_fit import get_Gs_Ts

def get_V_at_T(inp, rto, ladder):
    rto_theo = deepcopy(rto)
    rto_theo.sammy_runDIR = f"{rto.sammy_runDIR}_theo"
    rto_theo.bayes = False
    rto_theo.keep_runDIR = False
    inp.particle_pair.resonance_ladder = ladder

    options = syndatOPT(sampleRES = False,sample_counting_noise = False,sampleTMP = False,sampleTNCS = False) 

    covariance_data_at_theory = []
    for exp, meas in zip(inp.experiments, inp.measurement_models):
        if meas is None:
            cov = {}
        else:
            pw_true = Syndat_Model.generate_true_experimental_objects(particle_pair=inp.particle_pair, sammyRTO=rto_theo, generate_pw_true_with_sammy=True, generative_experimental_model=exp)
            raw_data = meas.generate_raw_data(pw_true, meas.model_parameters, options)
            _, cov, _ = meas.reduce_raw_data(raw_data,  options)
        covariance_data_at_theory.append(cov)

    Ds, covs = get_Ds_Vs(inp.datasets, covariance_data_at_theory, normalization_uncertainty=0.038)
    del rto_theo

    return scipy.linalg.block_diag(*covs)


def fit(rto, 
        starting_ladder, 
        external_resonance_indices, 
        particle_pair,
        D, V, experiments, datasets, covariance_data,

        V_is_inv = False,
        V_at_theory = False,
        measurement_models = None,
        V_projection = None,

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
                                "alpha":0.7},
        ):

    ### Check inputs
    assert alpha >0, "learn_rate 'alpha' must be greater than zero"

    if V_at_theory:
        assert measurement_models is not None, "User provided V_at_theory==True but did not provide measurement models"
        inp = sammy_classes.SammyInputDataYW(particle_pair=particle_pair, resonance_ladder=pd.DataFrame(), experiments=experiments, datasets=datasets, experimental_covariance=covariance_data, measurement_models=measurement_models)
        if V_projection is None:
            do_V_projection = False
        else:
            do_V_projection = True
            assert V_is_inv is True, "V projection is currently only setup on Vinv"


    saved_res_lads = []
    save_Pu = []
    saved_pw_lists = []
    saved_gradients = []
    chi2_log = []
    obj_log = []

    Pu_next, iE, igg, ign, iext, i_no_step = get_Pu_vec_and_indices(starting_ladder, particle_pair, external_resonance_indices)
    # print(f"Stepping until convergence\nchi2 values\nstep alpha: {[exp.title for exp in experiments]+['sum', 'sum/ndat']}")
    print(f"Stepping until convergence\nstep\talpha\t:\tobj\tchi2\n")
    for istep in range(steps):

        # Get current location derivatives and objective function values
        if V_at_theory:
            V = get_V_at_T(inp, rto, starting_ladder)
            if V_is_inv:
                V = np.linalg.inv(V)
            if do_V_projection:
                V = V_projection @ V @ V_projection

        chi2, dchi2_dpar_next, hessian_approx, sammy_pws, res_lad = evaluate_chi2_location_and_gradient(rto, Pu_next, starting_ladder, particle_pair, D, V, datasets,covariance_data,experiments, V_is_inv=V_is_inv)
        reg_pen, dreg_dpar_next = get_regularization_location_and_gradient(Pu_next, ign, iE, iext,
                                                                    lasso =lasso, lasso_parameters = lasso_parameters,
                                                                    ridge = ridge, ridge_parameters = ridge_parameters,
                                                                    elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)
        obj = chi2 + reg_pen
        
        if istep > 0:
            
            if LevMar:
                assert(LevMarV>1)
                if obj < obj_log[istep-1]:
                    alpha *= LevMarV
                    alpha = min(alpha,maxV)
                else:
                    if print_bool:
                        # print(f"Repeat step {int(istep)}, \talpha: {[exp.title for exp in experiments]+['sum', 'sum/ndat']}")
                        print(f"Repeat step {int(istep)}, \talpha: objective\tchi2")
                        print(f"\t\t{np.round(float(alpha),8):<10}: {obj:.2f}\t{chi2:.2f}")
                    while True:  
                        alpha /= LevMarVd
                        alpha = max(alpha, minV)
                        Pu_temp = take_step(Pu, alpha, dchi2_dpar, hessian_approx, dreg_dpar, iE, ign, gaus_newton=gaus_newton)

                        if V_at_theory:
                            V = get_V_at_T(inp, rto, starting_ladder)
                            if V_is_inv:
                                V = np.linalg.inv(V)
                            if do_V_projection:
                                V = V_projection @ V @ V_projection
                                
                        chi2_temp, dchi2_dpar_temp, hessian_approx_temp, sammy_pws_temp, res_lad_temp = evaluate_chi2_location_and_gradient(rto, Pu_temp, starting_ladder, particle_pair,D, V, datasets,covariance_data,experiments, V_is_inv=V_is_inv)
                        reg_pen_temp, dreg_dpar_temp = get_regularization_location_and_gradient(Pu_temp, ign, iE, iext,
                                                                                                lasso =lasso, lasso_parameters = lasso_parameters,
                                                                                                ridge = ridge, ridge_parameters = ridge_parameters,
                                                                                                elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)
                        obj_temp = chi2_temp + reg_pen_temp

                        if print_bool:
                            print(f"\t\t{np.round(float(alpha),8):<10}: {obj_temp:.2f}\t{chi2_temp:.2f}")
                        if obj_temp < obj_log[istep-1] or alpha==minV:
                            obj, chi2, dchi2_dpar_next, hessian_approx, dreg_dpar_next, sammy_pws, res_lad = obj_temp, chi2_temp, dchi2_dpar_temp, hessian_approx_temp, dreg_dpar_temp, sammy_pws_temp, res_lad_temp
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
                        criteria = f"Alpha below minimum value, taking solution {istep-1}"
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
            print(f"{int(istep)}\t{np.round(float(alpha),7):<8}:\t{obj:.2f}\t{chi2:.2f}")

        ### update Pu to Pu_next and save things
        Pu = Pu_next
        dchi2_dpar = dchi2_dpar_next
        dreg_dpar = dreg_dpar_next
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


def sgd(rto, 
        starting_ladder, 
        external_resonance_indices, 
        particle_pair,
        D, V, experiments, datasets, covariance_data,

        steps = 100, 
        patience = 10,
        batch_size = 25, 

        alpha = 1e-3, 
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-8,

        print_bool = True, 

        # batch_components = False,
        # gaus_newton = False, 
        
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
    # chi2_validation = []

    Pu_next, iE, igg, ign, iext, i_no_step = get_Pu_vec_and_indices(starting_ladder, particle_pair, external_resonance_indices)
    

    ### Initialize ADAM parameters
    t = 0
    mt = np.zeros_like(Pu_next)
    vt = np.zeros_like(Pu_next)
    best_obj = float('inf')
    ibest_obj = 0

    N = V.shape[0]
    U, s, Vt = np.linalg.svd(V, full_matrices=False)


    for istep in range(steps):
        
        #### Need to split regularization gradient into batches too
        dreg_dpar = np.zeros_like(Pu_next)
        reg_pen = 0 ### for now no regularization
        # reg_pen_temp, dreg_dpar_temp = get_regularization_location_and_gradient(Pu_next, ign, iE, iext,
        #                                                                         lasso =lasso, lasso_parameters = lasso_parameters,
        #                                                                         ridge = ridge, ridge_parameters = ridge_parameters,
        #                                                                         elastic_net = elastic_net, elastic_net_parameters = elastic_net_parameters)

        # for i in range(int(N/batch_size)):
        Pu = Pu_next
        t += 1
        
        # random indices of components
        index = np.random.choice(N, size=batch_size, replace=False) 
        s_i = s[index]; U_i = U[:, index]; Vt_i = Vt[index, :]
        # anti_index = np.array([i for i in range(N) if i not in index])
        # s_ai = s[anti_index]; U_ai = U[:, anti_index]; Vt_ai = Vt[anti_index, :]

        # Get derivatives and theory
        res_lad = get_p_resonance_ladder_from_Pu_vector(Pu, starting_ladder, particle_pair)
        Gs, Ts, sammy_pws = get_Gs_Ts(res_lad, particle_pair, experiments, datasets, covariance_data, rto)
        G = np.concatenate(Gs, axis=0); T = np.concatenate(Ts)
        G = zero_G_at_no_vary(G, res_lad)

        ## calculate chi2 and derivative
        chi2 = (D-T).T @ U @ np.diag(1/s) @ Vt @ (D-T)
        # chi2_validation.append((D-T).T @ U_ai @ np.diag(1/s_ai) @ Vt_ai @ (D-T))
        obj = chi2 + reg_pen

        ### check convergence and print
        if print_bool:
            print(f"{int(t)}\t:\t{obj:.2f}\t{chi2:.2f}")
        if obj < best_obj:
            best_obj = obj
            ibest_obj = t-1
            convergence_counter = 0
        else:
            convergence_counter += 1
        if convergence_counter >= patience:
            print(f"Early stopping: total objective function has not improved over {patience} steps")
            break

        dchi2_dpar =  - 2 * G.T@U_i @ np.diag(1/s_i) @ Vt_i@(D - T)
        hessian_approx =  G.T@U_i @ np.diag(1/s_i) @Vt_i@G
        # dchi2_dpar = np.clip(dchi2_dpar, -100, 100) ### Gradient cliping (value, could also do norm) for exploding gradients and numerical stability

        dObj_dtheta = dchi2_dpar + dreg_dpar
        mt = (beta_1 * mt + (1 - beta_1) * dObj_dtheta)
        vt = (beta_2 * vt + (1 - beta_2) * dObj_dtheta**2) 
        mt_hat = (mt / (1 - beta_1**t))
        vt_hat = (vt/ (1 - beta_2**t))
        
        Pu_next = (Pu - (alpha * mt_hat) / (np.sqrt(vt_hat) + epsilon))

        obj_log.append(obj)
        chi2_log.append(chi2)
        saved_gradients.append(dchi2_dpar)
        save_Pu.append(Pu)
        saved_pw_lists.append(sammy_pws)
        saved_res_lads.append(copy(res_lad))

        # print(f"Epoch {istep} done")
        

    return saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log, ibest_obj



from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputDataEXT
import scipy

def get_chi2(D, Vinv, pw_lists, experiments):
    Ts = []
    for pw, exp in zip(pw_lists, experiments):
        if exp.reaction == "transmission":
            key = "theo_trans"
        else:
            key = "theo_xs"
        Ts.append(pw[key].values)
    T = np.concatenate(Ts)
    return (D-T).T @ Vinv @ (D-T)


def run_sammy_EXT(sammyINP:SammyInputDataEXT, sammyRTO:SammyRunTimeOptions):
    
    rto_temp = deepcopy(sammyRTO)
    rto_temp.bayes = False
    
    ### get prior
    inpyw = sammy_classes.SammyInputDataYW(particle_pair=sammyINP.particle_pair, 
                                           resonance_ladder=sammyINP.resonance_ladder,  
                                           datasets=sammyINP.datasets, 
                                           experiments=sammyINP.experiments, 
                                           experimental_covariance=sammyINP.experimental_covariance,
                                           idc_at_theory=sammyINP.idc_at_theory,
                                           measurement_models=sammyINP.measurement_models)
    sammyOUT = run_sammy_YW(inpyw, rto_temp)

    if sammyINP.V_is_inv and not sammyINP.idc_at_theory:
        assert sammyINP.Vinv is not None
        assert(sammyINP.D is not None)
        assert(sammyINP.remove_V is False)
        V = sammyINP.Vinv
        D = sammyINP.D
        chi2 = get_chi2(D, sammyINP.Vinv, sammyOUT.pw, sammyINP.experiments_no_pup)
        sammyOUT.chi2 = chi2
        sammyOUT.chi2n = chi2/np.sum([len(each) for each in sammyOUT.pw])
    else:
        Ds, covs = get_Ds_Vs(sammyINP.datasets, 
            sammyINP.experimental_covariance, 
            normalization_uncertainty=sammyINP.cap_norm_unc)
        D = np.concatenate(Ds)
    
        if sammyINP.remove_V:
            V = np.diag(np.ones(len(D)))
        else:
            V = scipy.linalg.block_diag(*covs)

    ### iterate to convergence externally
    if sammyRTO.bayes:

        if sammyINP.minibatch:
            saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log, ibest = sgd(sammyRTO, sammyINP.resonance_ladder, sammyINP.external_resonance_indices, sammyINP.particle_pair, D, V, 
                                                                                                    sammyINP.experiments_no_pup, sammyINP.datasets, sammyINP.experimental_covariance,
                                                                                                    sammyINP.max_steps, sammyINP.patience, sammyINP.batch_size,
                                                                                                    sammyINP.alpha, sammyINP.beta_1, sammyINP.beta_2, sammyINP.epsilon, sammyRTO.Print)
            inpyw_post = sammy_classes.SammyInputDataYW(particle_pair=sammyINP.particle_pair, resonance_ladder=saved_res_lads[ibest],  
                                                        datasets=sammyINP.datasets, experiments=sammyINP.experiments, experimental_covariance=sammyINP.experimental_covariance,
                                                        max_steps=sammyINP.max_steps, step_threshold=sammyINP.step_threshold,
                                                        LevMar = sammyINP.LevMar,LevMarV = sammyINP.LevMarV,LevMarVd = sammyINP.LevMarVd,minF = sammyINP.minF,maxF = sammyINP.maxF,
                                                        initial_parameter_uncertainty=0.1, iterations=5)
            sammyOUT_post = run_sammy_YW(inpyw_post, sammyRTO)
            sammyOUT.pw_post = sammyOUT_post.pw_post
            sammyOUT.par_post = sammyOUT_post.par_post
            sammyOUT.chi2_post = sammyOUT_post.chi2_post
            sammyOUT.chi2n_post = sammyOUT_post.chi2n_post
            sammyOUT.Pu = save_Pu[0]
            sammyOUT.Pu_post = save_Pu[-1]


        else:
            saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log = fit(sammyRTO, sammyINP.resonance_ladder, sammyINP.external_resonance_indices, sammyINP.particle_pair, D, V, 
                                                                                            sammyINP.experiments_no_pup, sammyINP.datasets, sammyINP.experimental_covariance, 
                                                                                            sammyINP.V_is_inv, sammyINP.idc_at_theory, sammyINP.measurement_models, sammyINP.V_projection,
                                                                                            sammyINP.max_steps, sammyINP.step_threshold, sammyINP.alpha, sammyRTO.Print, sammyINP.gaus_newton, 
                                                                                            sammyINP.LevMar,sammyINP.LevMarV,sammyINP.LevMarVd,sammyINP.maxF,sammyINP.minF,
                                                                                            sammyINP.lasso, sammyINP.lasso_parameters,
                                                                                            sammyINP.ridge, sammyINP.ridge_parameters,
                                                                                            sammyINP.elastic_net, sammyINP.elastic_net_parameters)
            
            inpyw_post = sammy_classes.SammyInputDataYW(particle_pair=sammyINP.particle_pair, resonance_ladder=saved_res_lads[-1],  
                                                        datasets=sammyINP.datasets, experiments=sammyINP.experiments, experimental_covariance=sammyINP.experimental_covariance,
                                                        idc_at_theory=sammyINP.idc_at_theory, measurement_models=sammyINP.measurement_models)
            sammyOUT_post = run_sammy_YW(inpyw_post, rto_temp)
            sammyOUT.pw_post = sammyOUT_post.pw
            sammyOUT.par_post = sammyOUT_post.par
            sammyOUT.Pu = save_Pu[0]
            sammyOUT.Pu_post = save_Pu[-1]


            if sammyINP.idc_at_theory:
                if sammyINP.V_projection is not None:
                    Ds, covs = get_Ds_Vs(sammyINP.datasets, sammyOUT_post.covariance_data_at_theory, normalization_uncertainty=sammyINP.cap_norm_unc)
                    V = scipy.linalg.block_diag(*covs)
                    if sammyINP.V_is_inv:
                        V = np.linalg.inv(V)
                    V = sammyINP.V_projection @ V @ sammyINP.V_projection

                    chi2 = get_chi2(D, V, sammyOUT_post.pw, sammyINP.experiments_no_pup)
                    sammyOUT.chi2_post = chi2
                    sammyOUT.chi2n_post = chi2/np.sum([len(each) for each in sammyOUT_post.pw])
                else:
                    pass # because chi2 from run_sammy_YW will already have updated covariance

            elif sammyINP.V_is_inv: # if V_is_inv and idc_at_theory is not true, then static Vinv provided must be used to calculate chi2
                chi2 = get_chi2(D, sammyINP.Vinv, sammyOUT_post.pw, sammyINP.experiments_no_pup)
                sammyOUT.chi2_post = chi2
                sammyOUT.chi2n_post = chi2/np.sum([len(each) for each in sammyOUT_post.pw])
            else:
                sammyOUT.chi2_post = sammyOUT_post.chi2
                sammyOUT.chi2n_post = sammyOUT_post.chi2n

    
    else:
        pass

    return sammyOUT