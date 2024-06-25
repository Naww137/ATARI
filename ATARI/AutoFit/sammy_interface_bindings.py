
from ATARI.sammy_interface import sammy_classes
from ATARI.AutoFit.initial_FB_solve import InitialFBOPT


def sammyINP_solver_factory(options:InitialFBOPT, particle_pair, resonance_ladder, datasets, experiments, experimental_covariance, external_resonance_indices):

    universal_attributes = ["max_steps","step_threshold","LevMar","LevMarV","LevMarVd"]#,"minF","maxF"]
    universal_kwargs = {attr: getattr(options, attr) for attr in universal_attributes}
    
    if options.solver == "YW":
        solver_attributes = ["iterations","step_threshold_lag","initial_parameter_uncertainty"]
        solver_kwargs = {attr: getattr(options, attr) for attr in solver_attributes}
        solver_kwargs['autoelim_threshold'] = None
        solver_kwargs['LS'] = False
        solver_kwargs['minF'] = 1e-5
        solver_kwargs['maxF'] = 2.0
    
        inp = sammy_classes.SammyInputDataYW(particle_pair, resonance_ladder, datasets, experiments,experimental_covariance,external_resonance_indices=external_resonance_indices,
                                             **universal_kwargs, **solver_kwargs)


    elif options.solver == "EXT":
        solver_attributes = ["alpha","gaus_newton","lasso","lasso_parameters","ridge","ridge_parameters","elastic_net","elastic_net_parameters"]
        solver_kwargs = {attr: getattr(options, attr) for attr in solver_attributes}

        inp = sammy_classes.SammyInputDataEXT(particle_pair, resonance_ladder, datasets, experiments,experimental_covariance,external_resonance_indices=external_resonance_indices,
                                        **universal_kwargs, **solver_kwargs)
    else:
        raise ValueError("Solver not recognized")

    return inp
