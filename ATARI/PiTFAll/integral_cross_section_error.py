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
    xs_est, xs_true = get_rxns(par_true, par_est, sammy_exe, Ta_pair, energy_range, temperature, template, ['total'])
    assert(np.all(xs_est.E == xs_true.E))
    def trans_error(n):
        T_est  = calculate_transmission(xs_est , n)
        T_true = calculate_transmission(xs_true, n)
        return 1 - T_est/T_true
    xs_est_avg  = integral_average(xs_est .E.to_numpy(), xs_est .total.to_numpy())
    xs_true_avg = integral_average(xs_true.E.to_numpy(), xs_true.total.to_numpy())
    integral_xs_error = xs_est_avg / xs_true_avg - 1.0
    # def trans_error_min(n):
    #     return -abs(trans_error(n))
    # res = minimize(trans_error_min, x0=(1e-2,), bounds=[(1e-5,1.0)])
    # n = res.x
    return trans_error(n), integral_xs_error