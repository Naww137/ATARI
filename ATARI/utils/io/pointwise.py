from dataclasses import dataclass
from typing import Protocol
from pandas import DataFrame, Series, merge
import numpy as np

from ATARI.theory.xs import SLBW
from ATARI.theory.experimental import trans_2_xs, xs_2_trans
from ATARI.syndat.particle_pair import Particle_Pair
# from ATARI.utils.io.parameters import Parameters
from ATARI.utils.io.parameters import ExperimentalParameters
from ATARI.utils.io.parameters import TheoreticalParameters


class TheoreticalInfo(Protocol):
    @property
    def particle_pair(self) -> Particle_Pair:
        ...
    @property
    def resonance_ladder(self) -> DataFrame:
        ...
class ExperimentalInfoCalcModel(Protocol):
    @property
    def n(self) -> float:
        ...

def calculate_model(E: Series, theoinfo: TheoreticalInfo, expinfo: ExperimentalInfoCalcModel):
    xs_tot, _, _ = SLBW(E, theoinfo.particle_pair, theoinfo.resonance_ladder)
    trans = xs_2_trans(xs_tot, expinfo.n)
    return xs_tot, trans
    


class ExperimentalInfo_get_exp_xs_data(Protocol):
    @property
    def n(self) -> float:
        ...
    @property
    def dn(self) -> float:
        ...
    @property
    def blackthreshold(self) -> float:
        ...

def get_exp_xs_data(df: DataFrame, cov: DataFrame, exp_parm: ExperimentalInfo_get_exp_xs_data):
    xs_exp, CovXS = trans_2_xs(df.exp_trans, exp_parm.n, exp_parm.dn, cov)
    index_0trans = np.argwhere(np.array(df.exp_trans < exp_parm.blackthreshold)).flatten() 
    xs_exp.iloc[index_0trans] = np.nan
    if CovXS is not None:
        CovXS[index_0trans, :] = np.nan
        CovXS[:, index_0trans] = np.nan
        CovXS[index_0trans, index_0trans] = np.nan
        CovXS = DataFrame(CovXS, columns=df.E, index=df.E)
        CovXS.index.name = None
        exp_xs_unc = np.sqrt(np.diag(CovXS))
    else:
        exp_xs_unc = np.ones(len(xs_exp))
    return xs_exp, CovXS, exp_xs_unc


@dataclass
class PointwiseContainer:
    exp: DataFrame
    fine: DataFrame

    def add_model(self, theoretical_parameters: TheoreticalParameters, experimental_parameters: ExperimentalParameters, label: str):
        self.fine[f'{label}_xs'], _ = calculate_model(self.fine.E, theoretical_parameters, experimental_parameters)
        self.exp[f'{label}_xs'], self.exp[f'{label}_trans'] = calculate_model(self.exp.E, theoretical_parameters, experimental_parameters)

    def add_experimental(self, exp_df: DataFrame, CovT: DataFrame, exp_parm: ExperimentalInfo_get_exp_xs_data):
        merge_keys = list(set(self.exp.columns).intersection(exp_df.columns))
        df = merge(self.exp, exp_df, on=merge_keys)
        
        df.sort_values('E', inplace=True)
        df.reset_index(inplace=True, drop=True)
        CovT.sort_index(axis='index', inplace=True)
        CovT.sort_index(axis='columns', inplace=True)

        xs_exp, CovXS, exp_xs_unc = get_exp_xs_data(df, CovT, exp_parm)

        df['exp_xs'] = xs_exp
        df['exp_xs_unc'] = exp_xs_unc
        self.exp = df
        self.CovT = CovT
        self.CovXS = CovXS
    


# # def test_add_experimental():

# import unittest

# class TestPointwiseContainer(unittest.TestCase):
#     def test_add_experimental(self):

#         PwConObj = PointwiseContainer(
#                 DataFrame({'E':[10,0]}),
#                 DataFrame({'E':[10,5,0]})
#                 )
#         PwConObj.add_experimental(
#             exp_df = DataFrame({'E':[10,0], 'exp_trans':[0.01, 0.8]}),
#             CovT = DataFrame({'10':[0,0],'0':[0,0]}, index=['10','0']),
#             exp_parm = type('TestClass', (), {'n': 1, 'dn': 0.0, 'blackthreshold': 0.1})()
#         )

#         self.assertTrue(    np.isnan( PwConObj.exp.exp_xs.loc[PwConObj.exp.E==10] ) )
#         self.assertTrue(    np.isnan( PwConObj.exp.exp_xs_unc.loc[PwConObj.exp.E==10] ) )
#         self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[0:10, 10])) )
#         self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[10, 0:10])) )

#         self.assertFalse(   np.isnan( PwConObj.exp.exp_xs.loc[PwConObj.exp.E==0] ) )
#         self.assertFalse(   np.isnan( PwConObj.CovXS.loc[0,0] ) )
