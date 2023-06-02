

from abc import ABC, abstractmethod 
from pandas import DataFrame, Series, merge
from typing import Protocol
import numpy as np

from ATARI.theory.xs import SLBW
from ATARI.utils.misc import fine_egrid 
from ATARI.theory.experimental import trans_2_xs, xs_2_trans
from ATARI.utils.stats import chi2_val

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.utils.io.theoretical_parameters import TheoreticalParameters
from ATARI.utils.io.experimental_parameters import ExperimentalParameters
import ATARI.utils.io.hdf5 as h5io



class calculate_model_theoinfo_protocol(Protocol):
    @property
    def particle_pair(self) -> Particle_Pair:
        ...
    @property
    def resonance_ladder(self) -> DataFrame:
        ...
        
class calculate_model_expinfo_protocol(Protocol):
    @property
    def n(self) -> float:
        ...
def calculate_model(E: Series, theoinfo: calculate_model_theoinfo_protocol, expinfo: calculate_model_expinfo_protocol):
    xs_tot, _, _ = SLBW(E, theoinfo.particle_pair, theoinfo.resonance_ladder)
    trans = xs_2_trans(xs_tot, expinfo.n)
    return xs_tot, trans
    

class exp_xs_data_protocol(Protocol):
    @property
    def n(self) -> float:
        ...
    @property
    def dn(self) -> float:
        ...
    @property
    def blackthreshold(self) -> float:
        ...
def get_exp_xs_data(df: DataFrame, cov: DataFrame, exp_parm: exp_xs_data_protocol):
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


class PointwiseContainer:

    def __init__(self):
        pass

    # setters for lite container
    def set_exp(self, exp: DataFrame) -> None:
        self.exp = exp
    def set_CovT(self, CovT: DataFrame) -> None:
        self.CovT = CovT
    def set_mem(self, mem: str) -> None:
        self.mem = mem
    # setters for full container
    def set_fine(self, fine: DataFrame) -> None:
        self.fine = fine
    def set_CovXS(self, CovT: DataFrame) -> None:
        self.CovT = CovT
    

    @property
    def exp_models(self) -> list:
        return ['_'.join(each.split('_')[0:-1]) for each in self.exp.columns]
    @property
    def fine_models(self) -> list:
        return ['_'.join(each.split('_')[0:-1]) for each in self.fine.columns]

    ### methods for adding data to pointwise container
    def add_model(self, theoretical_parameters: TheoreticalParameters, experimental_parameters: ExperimentalParameters, overwrite=False):
        if theoretical_parameters.label in self.exp_models and not overwrite:
            print(f"Model '{theoretical_parameters.label}' already exists in pw.exp, bypassing pointwise reconstruction")
        else:
            self.exp[f'{theoretical_parameters.label}_xs'], self.exp[f'{theoretical_parameters.label}_trans'] = calculate_model(self.exp.E, theoretical_parameters, experimental_parameters)
        if self.mem == 'full':
            if theoretical_parameters.label in self.fine_models and not overwrite:
                print(f"Model '{theoretical_parameters.label}' already exists in pw.fine, bypassing pointwise reconstruction")
            else:
                self.fine[f'{theoretical_parameters.label}_xs'], _ = calculate_model(self.fine.E, theoretical_parameters, experimental_parameters)


    # # TODO: add a filter to catch warning for np.sqrt(negative transmission)
    def add_experimental(self, exp_df: DataFrame, CovT: DataFrame):
        merge_keys = list(set(self.exp.columns).intersection(exp_df.columns))
        df = merge(self.exp, exp_df, on=merge_keys)
        
        df.sort_values('E', inplace=True)
        df.reset_index(inplace=True, drop=True)
        CovT.sort_index(axis='index', inplace=True)
        CovT.sort_index(axis='columns', inplace=True)

        self.exp = df
        self.CovT = CovT
    
    def fill_exp_xs(self, exp_parm: exp_xs_data_protocol):
        xs_exp, CovXS, exp_xs_unc = get_exp_xs_data(self.exp, self.CovT, exp_parm)
        self.exp['exp_xs'] = xs_exp
        self.exp['exp_xs_unc'] = exp_xs_unc
        self.CovXS = CovXS

    def mem_2_lite(self):
        #TODO: Implement
        pass
    
    ### methods for writing the data out
    def to_hdf5(self, file: str, isample: int) -> None:
        h5io.write_pw_exp(file, isample, self.exp, self.CovT)
        if self.mem == 'full':
            h5io.write_pw_fine(file, isample, self.fine)


    ### methods for calculating FoMs
    # def chi2_trans(self, model_label) -> float:
    #     return chi2_val(self.exp[model_label], self.exp.exp_trans, self.exp.CovT)



### Builder classes

class BuildPointwiseContainer(ABC):
    def build_exp(self): pass
    def build_CovT(self): pass
    def build_mem(self, mem): pass
    def build_fine(self): pass
    def construct_full(self): pass
    def construct_lite(self): pass
    def construct_lite_w_CovT(self): pass

    @property
    @abstractmethod
    def product(self): 
        pass
    

class BuildPointwiseContainer_fromATARI(BuildPointwiseContainer):

    def __init__(self, exp: DataFrame, CovT: DataFrame = DataFrame(), ppeV: int = 100) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.exp = exp
        self.CovT = CovT
        self.ppeV = ppeV

    def reset(self) -> None:
        self._product = PointwiseContainer()

    @property
    def product(self) -> PointwiseContainer:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product
    
    def build_exp(self) -> None:
        self._product.set_exp(self.exp)
    def build_CovT(self) -> None:
        self._product.set_CovT(self.CovT)
    def build_mem(self,mem) -> None:
        self._product.set_mem(mem)
    def build_fine(self) -> None:
        self._product.set_fine(DataFrame({'E':fine_egrid(self.exp.E,self.ppeV)}))


    ### Construction methods to bypass the director

    def construct_full(self) -> PointwiseContainer:
        self.build_exp()
        self.build_CovT()
        self.build_fine()
        self.build_mem('full')
        return self.product
    
    def construct_lite(self) -> PointwiseContainer:
        self.build_exp()
        self.build_mem('lite')
        return self.product

    def construct_lite_w_CovT(self) -> PointwiseContainer:
        self.build_exp()
        self.build_CovT()
        self.build_mem('lite')
        return self.product




class BuildPointwiseContainer_fromHDF5(BuildPointwiseContainer):

    def __init__(self, hdf5_file: str, isample: int) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.hdf5_file = hdf5_file
        self.isample = isample

    def reset(self) -> None:
        self._product = PointwiseContainer()

    @property
    def product(self) -> PointwiseContainer:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product
    
    def build_exp(self) -> None:
        df, _ = h5io.read_pw_exp(self.hdf5_file, self.isample)
        if isinstance(df, Series):
            df = df.to_frame().T
        self._product.set_exp(df)

    def build_CovT(self) -> None:
        _, cov = h5io.read_pw_exp(self.hdf5_file, self.isample)
        if isinstance(cov, Series):
            cov = cov.to_frame().T
        self._product.set_CovT(cov)
    
    def build_mem(self,mem) -> None:
        self._product.set_mem(mem)
    
    def build_fine(self) -> None:
        df = h5io.read_pw_fine(self.hdf5_file, self.isample)
        if isinstance(df, Series):
            df = df.to_frame().T
        self._product.set_fine(df)


    ### Construction methods acting as director
    
    def construct_full(self) -> PointwiseContainer:
        self.build_exp()
        self.build_CovT()
        self.build_fine()
        self.build_mem('full')
        return self.product
    
    def construct_lite(self) -> PointwiseContainer:
        self.build_exp()
        self.build_mem('lite')
        return self.product

    def construct_lite_w_CovT(self) -> PointwiseContainer:
        self.build_exp()
        self.build_CovT()
        self.build_mem('lite')
        return self.product



### Director class
class DirectPointwiseContainer:
    """ The Director is only responsible for executing the building steps in a particular sequence. """

    def __init__(self) -> None:
        pass 
        #self._builder = None

    @property
    def builder(self) -> BuildPointwiseContainer:
        return self._builder

    @builder.setter
    def builder(self, builder: BuildPointwiseContainer) -> None:
        """The Director works with any builder instance that the client code passes to it."""
        self._builder = builder

    """The Director can construct several product variations using the same building steps."""
    
    def build_lite(self) -> None:
        self.builder.build_exp()
        self.builder.build_mem('lite')

    def build_lite_w_CovT(self) -> None:
        self.builder.build_exp()
        self.builder.build_CovT()
        self.builder.build_mem('lite')
    
    def build_full(self) -> None:
        self.builder.build_exp()
        self.builder.build_CovT()
        self.builder.build_fine()
        self.builder.build_mem('full')
