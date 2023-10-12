import numpy as np
from scipy.stats.distributions import chi2


def getD(xi,res_par_avg):    
    return res_par_avg['<D>']*2*np.sqrt(np.log(1/(1-xi))/np.pi)

def make_res_par_avg(J_ID, D_avg, Gn_avg, n_dof, Gg_avg, g_dof, print):

    res_par_avg = {'J_ID':J_ID, '<D>':D_avg, '<Gn>':Gn_avg, 'n_dof':n_dof, '<Gg>':Gg_avg, 'g_dof':g_dof}

    res_par_avg['D01']  = getD(0.01,res_par_avg)
    res_par_avg['D99']  = getD(0.99,res_par_avg)
    res_par_avg['Gn01'] = res_par_avg['<Gn>']*chi2.ppf(0.01, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gn99'] = res_par_avg['<Gn>']*chi2.ppf(0.99, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gg01'] = res_par_avg['<Gg>']*chi2.ppf(0.01, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gg99'] = res_par_avg['<Gg>']*chi2.ppf(0.99, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gt01'] = res_par_avg['Gn01'] + res_par_avg['Gg01']
    res_par_avg['Gt99'] = res_par_avg['Gn99'] + res_par_avg['Gg99']

    if print:
        print('D99  =',res_par_avg['D99'])
        print('Gn01 =',res_par_avg['Gn01'])
        print('Gn99 =',res_par_avg['Gn99'])
        print('Gg01 =',res_par_avg['Gg01'])
        print('Gg99 =',res_par_avg['Gg99'])
        print('Gt01 =',res_par_avg['Gt01'])
        print('Gt99 =',res_par_avg['Gt99'])

    return res_par_avg