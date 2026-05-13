import numpy as np
from scipy.optimize import minimize

from ATARI.PiTFAll.fnorm import get_rxns

def integral_average(x, y):
    return np.trapz(y, x)/(max(x)-min(x))

def calculate_transmission(xs, n):
    xs_tot = xs.total.to_numpy()
    E = xs.E
    T_avg = integral_average(E, np.exp(-n*xs_tot))
    return T_avg

def calculate_max_transmission_error(par_true, par_est, sammy_exe, Ta_pair, energy_range, temperature, template, n:float=0.01):
    xs_est, xs_true = get_rxns(par_true, par_est, sammy_exe, Ta_pair, energy_range, temperature, template, ['total', 'capture'])
    for rxn in ['total', 'capture']:
        # print(f'\nEst {rxn}:')
        # print(np.array(xs_est[rxn]))
        # print(f'\nTrue {rxn}:')
        # print(np.array(xs_true[rxn]))
        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)
        # print(f'\nDiff {rxn}:')
        # print(np.array(xs_est[rxn])+np.array(xs_true[rxn])*1j)
    assert(np.all(xs_est.E == xs_true.E))
    def trans_error(n):
        T_est  = calculate_transmission(xs_est , n)
        T_true = calculate_transmission(xs_true, n)
        return 1 - T_est/T_true
    xs_tot_est_avg  = integral_average(xs_est .E.to_numpy(), xs_est .total.to_numpy())
    xs_tot_true_avg = integral_average(xs_true.E.to_numpy(), xs_true.total.to_numpy())
    # print('xs_tot_est_avg', xs_tot_est_avg)
    # print('xs_tot_true_avg', xs_tot_true_avg)
    integral_tot_xs_error = xs_tot_est_avg / xs_tot_true_avg - 1.0

    xs_cap_est_avg  = integral_average(xs_est .E.to_numpy(), xs_est .capture.to_numpy())
    xs_cap_true_avg = integral_average(xs_true.E.to_numpy(), xs_true.capture.to_numpy())
    # print('xs_cap_est_avg', xs_cap_est_avg)
    # print('xs_cap_true_avg', xs_cap_true_avg)
    integral_cap_xs_error = xs_cap_est_avg / xs_cap_true_avg - 1.0
    # def trans_error_min(n):
    #     return -abs(trans_error(n))
    # res = minimize(trans_error_min, x0=(1e-2,), bounds=[(1e-5,1.0)])
    # n = res.x
    return trans_error(n), integral_tot_xs_error, integral_cap_xs_error