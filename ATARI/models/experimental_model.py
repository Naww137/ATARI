import numpy as np
from ATARI.utils.atario import update_dict
from ATARI.theory.experimental import e_to_t, t_to_e



class experimental_model:
    
    def __init__(self, 
                 title,
                 reaction,
                 energy_range,
                 energy_grid=None,
                 inputs={}, 
                 parameters={}, 
                 channel_width_info={},
                 additional_resfunc_lines=[])       -> None:
        
        self.title = title
        self.energy_range = energy_range
        self.reaction = reaction
        self.additional_resfunc_lines = additional_resfunc_lines

        ### default misc inputs
        if reaction == 'capture':
            default_inputs = {
               'alphanumeric'       :   ["USE MULTIPLE SCATTERING", 
                                         "INFINITE SLAB", 
                                         "NORMALIZE AS YIELD Rather than cross section", 
                                         "BROADENING IS WANTED", 
                                         "DO NOT SHIFT RPI RESOLUTION"],

               'ResFunc'            :   "RPI C"
            }
        elif reaction == 'transmission':
            default_inputs = {
               'alphanumeric'       :   ["BROADENING IS WANTED", 
                                         "DO NOT SHIFT RPI RESOLUTION"],

               'ResFunc'            :   "RPI T"
            }
        else:
            default_inputs = {
               'alphanumeric'       :   ["BROADENING IS WANTED"],

               'ResFunc'            :   ""
            }

        ### Default experiment parameters
        default_parameters = {
                        'n'         :   (0.067166, 0.0) ,
                        'FP'        :   (35.185, 0.0)   ,
                        't0'        :   (3326.0, 0.0)    ,
                        'burst'     :   (10, 1.0)       ,

                        'temp'      :   (300, 0.0)                 }
        
        ### default channel width info
        default_channel_width_info = {
                    "maxE": [np.max(energy_range)], 
                    "chw": [100.0],
                    "dchw": [0.8]
        }

        ### redefine dictionaries with supplied values
        self.inputs = update_dict(default_inputs, inputs) 
        self.parameters = update_dict(default_parameters, parameters)
        self.channel_width_info = update_dict(default_channel_width_info, channel_width_info)

        ### define energy grid
        if energy_grid is None:
            maxE, chw, dchw = [self.channel_width_info[key] for key in ["maxE", "chw", "dchw"]]
            energy_grid = np.array([])
            for i in range(len(maxE)):
                if i == 0:
                    tof_min_max = e_to_t(   np.array([min(self.energy_range), maxE[i]]), 
                                            self.parameters["FP"][0], True)*1e9 + self.parameters["t0"][0]
                else:
                    tof_min_max = e_to_t(   np.array([maxE[i-1], maxE[i]]), 
                                            self.parameters["FP"][0], True)*1e9 + self.parameters["t0"][0]
                    
                tof_grid = np.arange(min(tof_min_max), max(tof_min_max), chw[i])
                E = t_to_e((tof_grid-self.parameters["t0"][0])*1e-9, self.parameters["FP"][0], True) 
                energy_grid = np.concatenate([energy_grid, np.flipud(E)])
        else:
            pass
        
        self.energy_grid = energy_grid




    def __repr__(self):
        str = f"inputs:\n{self.inputs}"
        str += f"\nparameters:\n{self.parameters}"
        str += f"\nchannel_width_info:\n{self.channel_width_info}"
        return str
    



    def get_resolution_function_lines(self):
        if self.inputs["ResFunc"] in ["RPI T", "RPI C"]:
            pass
        else:
            if self.additional_resfunc_lines is None:
                raise ValueError("Need additional resolutions function lines if not using defaults")
            
        burst_line = f"BURST 0   {float(self.parameters['burst'][0]):<9} {float(self.parameters['burst'][1]):<9}"

        # need to actually loop over bin widths
        chann_lines = []
        maxE, chw, dchw = [self.channel_width_info[key] for key in ["maxE", "chw", "dchw"]]
        for e, c, dc in zip(maxE, chw, dchw):
            chann_lines.append(f"CHANN 0   {float(e):<9} {float(c):<9} {float(dc):<9}")


        return [burst_line]+self.additional_resfunc_lines+chann_lines