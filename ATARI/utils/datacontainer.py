import numpy as np
import pandas as pd

from ATARI.utils.atario import fill_resonance_ladder
from ATARI.utils.misc import fine_egrid
from ATARI.theory.experimental import trans_2_xs, xs_2_trans
from ATARI.theory.xs import SLBW
from ATARI.utils.misc import calc_xs_on_fine_egrid




class DataContainer():
    """
    ATARI Data Container object to hold theoretical, experimental, and estimated data.

    The DataContainer class is used to hold all necessary data for a given data realization. 
    A data realization is a single realization of theoretical resonance parameter and a single set experimental measurement data.
    Any number of parameter estimates may be added to the data container under different names.

    The DataContainer is instantiated without any inputs. 
    Then, theoretical, experimental, and estimate data can be added as needed via the 'add' methods.
    The function add_experimental takes the ATARI.syndat Experiment object as an input.
    Because the Syndat Experiment object also holds theoretical data, 

    The 'fill' method will calculate pointwise data depending on what has been added to the container. 
    The pointwise data will be on both an experimental grid and a fine grid and stored as attributes that can be easily accessed.
    The 'empty' method will erase all pointwise data calculated using the fill method in order to make the container more lightweight.
    """

    def __init__(self):
        self.has_theo = False
        self.has_exp = False
        self.has_est = False

        self.pw_fine = None
        self.pw_exp = None
        self.CovT = None
        self.CovXS = None

        return


    def add_theoretical(self, particle_pair, resonance_ladder):
        self.has_theo = True

        self.particle_pair = particle_pair
        self.theo_resonance_ladder = fill_resonance_ladder(resonance_ladder, particle_pair)

        # fill to automatically get pw_fine and cross section data
        self.fill()
        return


    def add_experimental(self, experiment, 
                                threshold=1e-2,
                                uncertainty_on_nan_xs = 100,
                                correlation_on_nan_xs = 0   ):
        self.has_exp = True
        self.threshold = threshold
        self.uncertainty_on_nan_xs = uncertainty_on_nan_xs 
        self.correlation_on_nan_xs = correlation_on_nan_xs 

        # copy experimental grid dataframe and some reduction parameters
        pw_exp = experiment.trans.copy(deep=True)
        CovT = experiment.CovT.copy(deep=True)
        pw_exp['theo_trans'] = experiment.theo.theo_trans

        self.pw_exp = pw_exp
        self.CovT = CovT
        self.n = experiment.redpar.val.n
        self.n_unc = experiment.redpar.unc.n
        
        # could add other experimental data or just carry around the entire experiment object or cary none of that info

        # run fill to automatically get pw_fine and cross section data
        self.fill()

        return


    def add_estimate(self, resonance_ladder, 
                                    est_name='est',
                                    particle_pair=None):
        if self.has_est:
            self.est_resonance_ladder[f'{est_name}'] = resonance_ladder
        else:
            self.est_resonance_ladder = {est_name:resonance_ladder}

        self.has_est = True
        self.fill()



    def fill(self):
        
        if self.has_theo:
            self.calculate_theo_pw()

        if self.has_exp:

            # convert experimental data to cross section and covariance too
            self.pw_exp.sort_values('E', inplace=True)
            self.CovT.sort_index(axis='index', inplace=True)
            self.CovT.sort_index(axis='columns', inplace=True)
            xs_exp, CovXS = trans_2_xs(self.pw_exp.exp_trans, self.n, self.n_unc, self.CovT)

            # set xs and Cov values below threshold to nan, Cov should be large variance 0 covaraiance
            index_0trans = np.argwhere(np.array(self.pw_exp.exp_trans<self.threshold)).flatten() 
            xs_exp.iloc[index_0trans] = np.nan
            CovXS[index_0trans, :] = self.correlation_on_nan_xs
            CovXS[:, index_0trans] = self.correlation_on_nan_xs
            CovXS[index_0trans, index_0trans] = self.uncertainty_on_nan_xs 
            self.pw_exp['exp_xs'] = xs_exp
            self.pw_exp['exp_xs_unc'] = np.sqrt(np.diag(CovXS))
            CovXS = pd.DataFrame(CovXS, columns=self.pw_exp.E, index=self.pw_exp.E)
            CovXS.index.name = None
            self.CovXS = CovXS
            
        if self.has_est:
            est_ladder_dict = self.est_resonance_ladder
            for estimate_key, estimate_ladder in est_ladder_dict.items():
                estimate_ladder = fill_resonance_ladder(estimate_ladder, self.particle_pair, J = 3.0,
                                                                                            chs = 1.0,
                                                                                            lwave=0.0,
                                                                                            J_ID=1.0)
                # experimental grid
                xs_tot, _, _ = SLBW(self.pw_exp.E, self.particle_pair, estimate_ladder)
                self.pw_exp[f'{estimate_key}_xs'] = xs_tot
                self.pw_exp[f'{estimate_key}_trans'] = xs_2_trans(xs_tot, self.n)
                # fine grid
                xs_tot, _, _ = SLBW(self.pw_fine.E, self.particle_pair, estimate_ladder)
                self.pw_fine[f'{estimate_key}_xs'] = xs_tot




    def empty(self):
        self.pw_fine = None
        self.CovXS = None

        return
    

    def calculate_theo_pw(self):
        if self.has_exp:
            minE = min(self.pw_exp.E)
            maxE = max(self.pw_exp.E)
            xs_tot, _, _ = SLBW(self.pw_exp.E, self.particle_pair, self.theo_resonance_ladder)
            self.pw_exp[f'theo_xs'] = xs_tot
            self.pw_exp[f'theo_trans'] = xs_2_trans(xs_tot, self.n)
        else:
            minE = min(self.theo_resonance_ladder.E) - self.theo_resonance_ladder.Gt*1e-3
            maxE = max(self.theo_resonance_ladder.E) + self.theo_resonance_ladder.Gt*1e-3
        fineE, theo_xs_tot = calc_xs_on_fine_egrid(np.array([minE, maxE]), 1e2, self.particle_pair, self.theo_resonance_ladder)
        self.pw_fine = pd.DataFrame({'E':fineE, 'theo_xs':theo_xs_tot})
        return

