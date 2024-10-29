import unittest
import numpy as np
import pandas as pd
import os
from ATARI.utils.atario import save_syndat_model, save_syndat_control, load_general_object
from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.control import Syndat_Control
from ATARI.syndat.data_classes import syndatOPT
from ATARI.ModelData.particle_pair import Particle_Pair

__doc__ == """

"""



class TestSyndatSaveLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synmod = Syndat_Model(options=syndatOPT(sampleRES=False))
        cls.syncon = Syndat_Control(Particle_Pair(), 
                                    [cls.synmod], 
                                    sampleRES=False)
        
        
        energy_grid = cls.synmod.generative_experimental_model.energy_grid
        df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.01,1.0,len(energy_grid))})
        # df_true.sort_values('E', ascending=True, inplace=True)
        cls.df_true = df_true

        cls.pickle_file_path = os.path.join(os.path.dirname(__file__), "test.pkl")


    def test_save_syndat_model_with_samples_directly(self):
        self.synmod.sample(pw_true=self.df_true)
        save_syndat_model(self.synmod, self.pickle_file_path, clear_samples=False)
        self.assertTrue(os.path.exists(self.pickle_file_path))
        os.remove(self.pickle_file_path)
        
    def test_save_syndat_model_with_clearing_samples_from_control(self):
        self.syncon.sample(pw_true_list=[self.df_true])
        save_syndat_model(self.synmod, self.pickle_file_path, clear_samples=True)
        self.assertTrue(os.path.exists(self.pickle_file_path))
        os.remove(self.pickle_file_path)
    
    # def test_save_model_without_clearing_samples_from_control(self):
    #     self.syncon.sample(pw_true_list=[self.df_true], num_samples=3)
    #     _ = self.syncon.get_sample(0)
    #     self.assertRaises(ValueError, save_syndat_model, syndat_model=self.synmod, path=self.pickle_file_path, clear_samples=False)

    def test_load_syndat_model_with_clearing_samples_from_control(self):
        self.syncon.sample(pw_true_list=[self.df_true])
        save_syndat_model(self.synmod, self.pickle_file_path, clear_samples=True)
        self.assertTrue(os.path.exists(self.pickle_file_path))
        syndat_loaded = load_general_object(self.pickle_file_path)
        os.remove(self.pickle_file_path)
        self.assertIsInstance(syndat_loaded, Syndat_Model)



if __name__ == '__main__':
    unittest.main()
