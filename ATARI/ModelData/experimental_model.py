import numpy as np
from ATARI.theory.experimental import e_to_t, t_to_e
from ATARI.ModelData.structuring import parameter


class Experimental_Model:

    n = parameter()
    FP = parameter()
    t0 = parameter()
    burst = parameter()
    temp = parameter()

    def __init__(self, **kwargs):
        """
        Experimental Model is a class that holds information 

        Parameters
        ----------
        **kwargs : dict, optional
            Any keyword arguments are used to set attributes on the instance.

        Attributes
        ----------
        title: str
            Title
        reaction: str
            Title
        energy_range: str
            Title
        template: str
            Title
        energy_grid: str
            Title
        n: str
            Title
        FP: str
            Title
        t0: str
            Title
        burst: str
            Title
        temp: str
            Title
        additional_resfunc_lines: str
            Title
        """
        self.title = "T12mm"
        self.reaction = "transmission"
        self.energy_range = [200, 250]
        self.template = None
        self.energy_grid = None

        self.n = (0.067166, 0.0)
        self.FP = (35.185, 0.0)
        self.t0 = (3326.0, 0.0)
        self.burst = (10, 1.0)
        self.temp = (300, 0.0)
        
        self.additional_resfunc_lines = []

        self.channel_widths = {
            "maxE": [np.max(self.energy_range)],
            "chw": [100.0],
            "dchw": [0.8]
        }

        # update kwargs to get user input for some parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # default misc inputs
        if self.reaction == 'capture':
            self.sammy_inputs = {
                'alphanumeric':   ["USE MULTIPLE SCATTERING",
                                   "INFINITE SLAB",
                                   "NORMALIZE AS YIELD Rather than cross section",
                                   "BROADENING IS WANTED",
                                   "DO NOT SHIFT RPI RESOLUTION"],

                'ResFunc':   "RPI C"
            }
        elif self.reaction == 'transmission':
            self.sammy_inputs = {
                'alphanumeric':   ["BROADENING IS WANTED",
                                   "DO NOT SHIFT RPI RESOLUTION"],

                'ResFunc':   "RPI T"
            }
        else:
            self.sammy_inputs = {
                'alphanumeric':   ["BROADENING IS WANTED"],

                'ResFunc':   ""
            }        
        
        # update kwargs again if user supplied sammy inputs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # catch error if no energy information is given at all and update energy range if grid is given
        if self.energy_range is None:
            if self.energy_grid is None:
                raise ValueError("Neither energy range or energy grid was given")
            else:
                self.energy_range = [np.min(self.energy_grid), np.max(self.energy_grid)]

        # define energy grid
        if self.energy_grid is None:

            if "energy_range" in kwargs.keys() and "channel_widths" not in kwargs.keys():
                print("WARNING: no energy grid or channel width information provided, using defaults")

            maxE, chw, dchw = [self.channel_widths[key] for key in ["maxE", "chw", "dchw"]]
            if max(maxE) < max(self.energy_range): print("WARNING: channel width maxE is less than max(energy_range)")

            self.energy_grid = np.array([])
            self.tof_grid = np.array([])
            for i in range(len(maxE)):
                if i == 0:
                    tof_min_max = e_to_t(np.array([min(self.energy_range), maxE[i]]),
                                         self.FP[0], True)*1e9 + self.t0[0]
                else:
                    tof_min_max = e_to_t(np.array([maxE[i-1], maxE[i]]),
                                         self.FP[0], True)*1e9 + self.t0[0]
                tof = np.arange(min(tof_min_max), max(tof_min_max), chw[i])
                E = t_to_e((tof-self.t0[0])*1e-9, self.FP[0], True)
                self.energy_grid = np.concatenate([self.energy_grid, np.flipud(E)])
                self.tof_grid = np.concatenate([self.tof_grid, np.flipud(tof)])
        else:
            self.energy_grid = np.sort(self.energy_grid)
            self.tof_grid = e_to_t(self.energy_grid, self.FP[0], True)*1e9 + self.t0[0]

        energy_range_mask = (self.energy_grid>=min(self.energy_range)) & (self.energy_grid<=max(self.energy_range))
        self.energy_grid = self.energy_grid[energy_range_mask]
        self.tof_grid = self.tof_grid[energy_range_mask]

    @property
    def title(self):
        'The name of the experiment'
        return self._title
    @title.setter
    def title(self, title):
        self._title = title

    @property
    def energy_range(self):
        'The energy range for the experiment'
        return self._energy_range
    @energy_range.setter
    def energy_range(self, energy_range):
        # TODO: truncate energy grid
        self._energy_range = energy_range

    @property
    def template(self):
        return self._template
    @template.setter
    def template(self, template):
        self._template = template

    @property
    def energy_grid(self):
        return self._energy_grid
    @energy_grid.setter
    def energy_grid(self, energy_grid):
        # TODO: check and truncate if outside of energy range
        self._energy_grid = energy_grid

    @property
    def reaction(self):
        'The type of reaction to model'
        return self._reaction
    @reaction.setter
    def reaction(self, reaction):
        self._reaction = reaction

    @property
    def channel_widths(self):
        return self._channel_widths
    @channel_widths.setter
    def channel_widths(self, channel_widths):
        self._channel_widths = channel_widths

    @property
    def sammy_inputs(self):
        return self._sammy_inputs
    @sammy_inputs.setter
    def sammy_inputs(self, sammy_inputs):
        self._sammy_inputs = sammy_inputs


    def __repr__(self):
        string = f"inputs:\n{self.sammy_inputs}"
        string += f"\nchannel_width_info:\n{self.channel_widths}"
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string


    def get_resolution_function_lines(self):
        if self.sammy_inputs["ResFunc"] in ["RPI T", "RPI C"]:
            pass
        else:
            if self.additional_resfunc_lines is None:
                raise ValueError("Need additional resolutions function lines if not using defaults")

        burst_line = f"BURST 0   {float(self.burst[0]):<9} {float(self.burst[1]):<9}"

        # need to actually loop over bin widths
        chann_lines = []
        maxE, chw, dchw = [self.channel_widths[key] for key in ["maxE", "chw", "dchw"]]
        for e, c, dc in zip(maxE, chw, dchw):
            chann_lines.append(f"CHANN 0   {float(e):<9} {float(c):<9} {float(dc):<9}")


        return [burst_line]+self.additional_resfunc_lines+chann_lines

    def truncate_energy_range(self,
                              new_energy_range):
        if min(new_energy_range) < min(self.energy_range):
            raise ValueError("new energy range is less than existing experimental_model.energy_range")
        if max(new_energy_range) > max(self.energy_range):
            raise ValueError("new energy range is more than existing experimental_model.energy_range")
        
        self.energy_range = new_energy_range
        self.energy_grid = self.energy_grid[(self.energy_grid>min(new_energy_range)) & (self.energy_grid<max(new_energy_range))]
        self.tof_grid =  e_to_t(self.energy_grid,self.FP[0], True)*1e9 + self.t0[0]
        return
