
import numpy as np
import pandas as pd
import os




class sample_case:

    def __init__(self, case_directory, icase):

        self.case_directory = case_directory
        self.icase = icase
    

    def analyze(self):
    
        # read pw data and compile into dataframe
        syndat_pw_df = pd.read_csv(os.path.join(self.case_directory, f'sample_{self.icase}', f'syndat_{self.icase}_pw.csv'))
        fit_pw_df = pd.read_csv(os.path.join(self.case_directory, f'sample_{self.icase}', f'fit_{self.icase}_pw.csv'))

        pw_data = syndat_pw_df.loc[:, ['E','theo_trans','exp_trans','exp_trans_unc']]
        pw_data['est_trans'] = fit_pw_df['est_trans']
        self.pw_data = pw_data

        # pull vectors from dfs and cast into numpy arrays
        exp_trans = np.array(syndat_pw_df.exp_trans)
        exp_trans_unc = np.array(syndat_pw_df.exp_trans_unc)
        est_trans = np.array(fit_pw_df.est_trans)
        theo_trans = np.array(syndat_pw_df.theo_trans)

        # calculate some integral FoMs and put into df
        dof = len(exp_trans)
        est_SE = np.sum((exp_trans-est_trans)**2)
        sol_SE = np.sum((exp_trans-theo_trans)**2)
        est_chi_square = np.sum((exp_trans-est_trans)**2/exp_trans_unc**2)
        sol_chi_square = np.sum((exp_trans-theo_trans)**2/exp_trans_unc**2)

        est_sol_SE = np.sum((theo_trans-est_trans)**2)

        FoMs = pd.DataFrame({'fit_exp'         :   [est_SE,     est_chi_square, est_chi_square/dof], 
                                'theo_exp'     :   [sol_SE,     sol_chi_square, sol_chi_square/dof],
                                'fit_theo'     :   [est_sol_SE, None,           None],
                                'FoM'          :   ['SE',      'Chi2',         'Chi2/dof']})
        FoMs.set_index('FoM', inplace=True)
        self.FoMs = FoMs





        # read parameter data and compile into dataframe
        syndat_par_df = pd.read_csv(os.path.join(self.case_directory, f'sample_{self.icase}', f'syndat_{self.icase}_par.csv'))
        fit_par_df = pd.read_csv(os.path.join(self.case_directory, f'sample_{self.icase}', f'fit_{self.icase}_par.csv'))
        syndat_par_df.sort_values('E', inplace=True)
        fit_par_df.sort_values('E', inplace=True)

        # calculate some FoMs WRT known solution and put into df
  
        