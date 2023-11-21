import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def XS_to_Yield(reaction_xs, total_xs, target_per_barn):
    """Calculates the theoretical yield and count rate distributions

    Arguments:
        capture_xs_theo -- (n) array of floats: The theoretical capture cros-section distribution for Energy
        total_xs_theo   -- (n) array of floats: The theoretical total cross-section distribution for Energy
        target_per_barn -- float:               The number of target atoms per unit length

    Returns:
        yield_theo      -- (n) array of floats: The theoretical capture yield distribtuion for Energy
    """
    Yield            = np.multiply(np.divide(reaction_xs[0],total_xs[0]), (1-np.exp(np.multiply(-target_per_barn,total_xs[0]))))
    Yield_uncertainty= np.zeros(len(Yield))
    
    return([Yield,Yield_uncertainty])



def Yield_to_Flux(countrate,background,Yield):
    """__Summary__
    
    Arugments:
        __Name__ -- __Type__: __Description__
    
    Modes:
        __Name__ -- __Type__: __Value 1__=__Mode Option 1__
                              __Value 2__=__Mode Option 2__
    
    Returns:
        __Name__ -- __Type__: __Description__
    """
    #Sample Calculation:
    flux= np.divide(countrate[0] - background[0], Yield[0])
    
    #Uncertainty Calculation:
    Partial_flux_Cc            = np.divide(1, Yield[0])
    Partial_flux_Cb            =-np.divide(1, Yield[0])
    Partial_flux_yield         =-np.divide(countrate[0] - background[0], np.power(Yield[0],2))
    
    flux_uncertainty = np.sqrt(np.power(Partial_flux_Cc    * countrate[1] ,2)
                              +np.power(Partial_flux_Cb    * background[1],2)
                              +np.power(Partial_flux_yield * Yield[1]     ,2))
    
    return([flux,flux_uncertainty])



def Countrate_to_Yield(countrate, background, flux, scaling):
    """Calculates sample yield and it's uncertainty.
    
    Arguments:
        count_exp       -- (2,n,m) array of floats: The experimental capture count rate distribution for Energy and it's uncertainty
        background_exp  -- (2,n,m) array of floats: The experimental background count rate distribtuion for Energy and it's uncertainty
        flux_exp        -- (2,n,m) array of floats: The experimental neutron flux distribtuion for Energy and it's uncertainty

    Returns:
        yield_exp       -- (2,n,m) array of floats: The experimental yield distribution for Energy and it's uncertainty
        
        !Note: The first index for each value is the experimental distribution, the second index is the associated uncertainty
    """
    #Distribution Calculations:
    Yield             = np.multiply(scaling[0], np.divide(countrate[0] - background[0], flux[0]))
    
    #Uncertainty Calculations:
    partial_Y_Cc      = np.divide(scaling[0], flux[0])
    partial_Y_Cb      =-np.divide(scaling[0], flux[0])
    partial_Y_flux    =-np.multiply(scaling[0], np.divide(countrate[0] - background[0], np.power(flux[0],2)))
    partial_Y_scaling = np.divide(countrate[0] - background[0], flux[0])
    
    Yield_uncertainty = np.sqrt(np.power(np.multiply(partial_Y_Cc     , countrate[1]) ,2)
                               +np.power(np.multiply(partial_Y_Cb     , background[1]),2)
                               +np.power(np.multiply(partial_Y_flux   , flux[1])      ,2)
                               +np.power(np.multiply(partial_Y_scaling, scaling[1])   ,2))
    
    return([Yield,Yield_uncertainty])

def Yield_to_Countrate(Yield, background, flux, scaling):
    """Calculates the theoretical yield and count rate distributions

    Arguments:
        yield_theo      -- (n) array of floats: The theoretical capture yield distribtuion for Energy
        background_theo -- (n) array of floats: The theoretical background distribution for Energy
        flux_theo       -- (n) array of floats: The theoretical neturon flux rate distribution for Energy

    Returns:
        count_theo      -- (n) array of floats: The theoretical capture count rate distribtuion for Energy
    """
    #Distribution Calculations:
    countrate             = np.divide(np.multiply(Yield[0],flux[0]),scaling[0]) + background[0]
    
    #Uncertainty Calculations:
    partial_Cc_Cb         = 1
    partial_Cc_Y          = np.divide(flux[0],  scaling[0])
    partial_Cc_flux       = np.divide(Yield[0], scaling[0])
    partial_Cc_scaling    = np.divide(np.multiply(flux[0], Yield[0]), np.power(scaling[0],2))
    
    countrate_uncertainty = np.sqrt(np.power(np.multiply(partial_Cc_Cb     , background[1]),2)
                                         +np.power(np.multiply(partial_Cc_Y      , Yield[1])     ,2)
                                         +np.power(np.multiply(partial_Cc_flux   , flux[1])      ,2)
                                         +np.power(np.multiply(partial_Cc_scaling, scaling[1])   ,2))
    
    return([countrate,countrate_uncertainty])



def Sample_Poisson_Distribution(Dis):
    """Generates a poisson distribution around a given theoretical distribution.
    
    Arguments:
        Theo_Dis        -- (n) array of floats:     The theoretical distribution to be sampled around

    Returns:
        Sampled_Dis     -- (2,n) array of floats: The noisy sampled distribtuion around the theoreical and the associated uncertainties
        
        !Note: The first index is the sampled distribution, the second index is the associated uncertainty
    """
    #Sample Calculations:
    sample_data= np.random.poisson(Dis[0])
    
    #Uncertainty Calculations:
    sample_data_uncertainty = np.sqrt(sample_data)
    
    return([sample_data,sample_data_uncertainty])

def Sample_Gaussian_Distribution(Dis):
    """Generates a poisson distribution around a given theoretical distribution.
    
    Arguments:
        Theo_Dis        -- (n) array of floats:     The theoretical distribution to be sampled around

    Returns:
        Sampled_Dis     -- (2,n) array of floats: The noisy sampled distribtuion around the theoreical and the associated uncertainties
        
        !Note: The first index is the sampled distribution, the second index is the associated uncertainty
    """
    #Sample Calculations:
    sample_data= np.random.normal(Dis[0],Dis[1])
    
    #Uncertainty Calculations:
    sample_data_uncertainty = Dis[1]
    
    return([sample_data,sample_data_uncertainty])