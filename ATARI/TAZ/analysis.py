import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt

__doc__ = """
This module is used to analyze the data created by TAZ with scores, confusion matrices, probability
distributions, etc.
"""

def __fractionEstimation(N:int, n:int):
    """
    A function used to estimate the fraction of successful counts when restricted between 0 and 1.
    The estimated fraction and standard deviation are derived from the expectation value and
    variance of the binomial probability distribution.

    Parameters:
    ----------
    N : int
        The total number of trials.

    n : int
        The number of successful results from the `N` trials.

    Returns:
    -------
    frac_est : float
        The estimated fraction of successful counts.

    frac_std : float
        The standard deviation on the estimated fraction.
    """

    frac_est = (n+1)/(N+2)
    frac_var = ((n+1)*(n+2))/((N+2)*(N+3)) - frac_est**2
    frac_std = np.sqrt(frac_var)
    return frac_est, frac_std

def spinGroupNames(num_groups:int, last_false:bool=True):
    """
    A function that provides default names for each of the `num_groups` spingroup. The spingroup
    names are given by letters, provided alphabetically, excluding "F", "I", and "O" to avoid
    confusion. If `last_false` is True, then the last group will have the name, "False".

    ...
    """

    # Error checking:
    if type(num_groups) == float:
        if (num_groups % 1.0 != 0.0) or (num_groups <= 0):
            raise ValueError(f'Only a positive integer number of spingroups are allowed, not {num_groups}.')
        else:
            num_groups = int(num_groups)
    if num_groups > 24:
        raise ValueError(f'You are implying you have {num_groups} groups. I do not have enough letters in the alphabet to differentiate each spingroup!')
    
    # Assigning spingroup names:
    letters = 'ABCDEGHJKLMNPQRSTUVWXYZ'
    if last_false:
        sg_names = [*letters[:num_groups-1], 'False']
    else:
        sg_names = [*letters[:num_groups]]
    return sg_names

def PrintScore(pred_probs:ndarray, answer:ndarray,
               run_name:str='', metric:str='best guess'):
    """
    ...
    """

    metric = metric.lower()
    if   metric == 'best guess':
        guess = np.argmax(pred_probs, axis=1)
        score = np.count_nonzero(answer == guess) / np.size(answer)
    elif metric == 'probability score':
        score = np.mean(pred_probs[:,answer])
    else:
        raise NotImplementedError(f'Unknown metric, "{metric}"')

    if run_name:    print(f'{run_name} score = {score:7.2%}')
    else:           print(f'Score = {score:7.2%}')
    return score

def PrintPredScore(pred_probs:ndarray,
                   run_name:str='', metric:str='best guess'):
    """
    ...
    """

    metric = metric.lower()
    if   metric == 'best guess':
        guess = np.argmax(pred_probs, axis=1)
        score = np.mean(pred_probs[:,guess])
    elif metric == 'probability score':
        score = np.mean(np.sum(pred_probs**2, axis=1))
    else:
        raise NotImplementedError(f'Unknown metric, "{metric}"')

    if run_name:    print(f'Predicted {run_name} score = {score:7.2%}')
    else:           print(f'Predicted score = {score:7.2%}')
    return score

def ConfusionMatrix(pred_probs:ndarray, answer:ndarray, sg_names:list=None):
    """
    Calculate and print the confusion matrix with proper labels.

    ...
    """

    num_groups = pred_probs.shape[1]
    pred_ids = np.argmax(pred_probs, axis=1)
    true_ids = answer
    
    # Ensure the input arrays have the same length:
    if len(pred_ids) != len(true_ids):
        raise ValueError("Input arrays must have the same length.")

    # Determine the number of groups:
    if sg_names is None:
        sg_names = spinGroupNames(num_groups)
    elif len(sg_names) != num_groups:
        raise ValueError('The length of the list of spingroup names must equal the number of spingroups, given by the rows of `pred_probs`.')

    # Initialize the confusion matrix:
    confusion_matrix = np.zeros((num_groups, num_groups), dtype=int)

    # Fill in the confusion matrix:
    for pred_id, true_id in zip(pred_ids, true_ids):
        confusion_matrix[pred_id, true_id] += 1

    # Print the confusion matrix:
    print("Confusion Matrix:")
    confusion_matrix_table = pd.DataFrame(confusion_matrix, index=sg_names, columns=sg_names)
    # confusion_matrix_table.index.name = "Predicted"
    # confusion_matrix_table.columns.name = "Correct"
    print(confusion_matrix_table)

    return confusion_matrix

def ProbCorrPlot(pred_probs:ndarray, answer:ndarray,
                 sg_names:list=None, fig_name:str='', fig_num:int=None):
    """
    ...
    """

    num_groups = pred_probs.shape[1]
    if sg_names is None:
        sg_names = spinGroupNames(num_groups)

    nBin = round(np.sqrt(len(answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    X = (edges[:-1] + edges[1:])/2 # the center of each bin
    for t in range(num_groups):
        prob_guess_type = pred_probs[:,t]
        prob_guess_cor  = prob_guess_type[answer == t]
        count_all = np.histogram(prob_guess_type, bins=edges)[0]
        count_cor = np.histogram(prob_guess_cor , bins=edges)[0]

        # Only plot the cases where counts exist:
        X2         = X[count_all != 0]
        count_all2 = count_all[count_all != 0]
        count_cor2 = count_cor[count_all != 0]

        # Estimating probability and confidence:
        prob_ans_cor_est, prob_ans_cor_std = __fractionEstimation(count_all2, count_cor2)

        # Plotting:
        plt.figure(fig_num)
        plt.clf()
        plt.errorbar(X2,prob_ans_cor_est, prob_ans_cor_std, capsize=3, ls='none', c='k')
        plt.scatter(X2,prob_ans_cor_est, marker='.', s=14, c='k')
        plt.plot([0,1], [0,1], ':b', lw=2, label='Ideal Correlation')

        # Probability density on the guessed probability:
        norm = 0.25/np.max(count_all)
        plt.fill_between(X,norm*count_all, alpha=0.8, color=(0.1,0.5,0.1), label='Probability Density')
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title(f'Probability Accuracy for Spin Group {sg_names[t]}', fontsize=18)
        plt.xlabel('Predicted Probability', fontsize=15)
        plt.ylabel('Fraction of Correct Assignments', fontsize=15)
        plt.legend(fontsize=10)

        if fig_name:
            fig_name = str(fig_name).format(sgn=sg_names[t])
            plt.savefig(f'{fig_name}.png')
        else:
            plt.show()

def ecdf(X, lb:float=None, ub:float=None,
         color:str='k', linestyle:str='-',
         density:bool=True, label:str=None, ax=None):
    """
    Plots the empirical cumulative distribution function (ECDF) of a float array, `X`.

    Parameters:
    ----------
    X         :: float, array-like
        Values to plot the emiprical CDF for.
    
    lb        :: float
        Optional lower bound.

    ub        :: float
        Optional upper bound.

    color     :: str
        The line color.

    linestyle :: str
        The line style.

    density   :: bool
        Determines whether cumulative densities are plotted (highest value is 1.0), or cumulative
        counts are plotted (highest value is the total number of values). Default is True.

    label     :: str
        The legend label for the line.
    
    ax        :: plt.axes
        Optional ax option. If not provided, `plt.gca()` is used.
    """

    if ax is None:
        ax = plt.gca()

    # Sort the data in ascending order
    X_sorted = np.sort(X)

    # Create an array of unique values and their corresponding cumulative probabilities
    X_unique, counts = np.unique(X_sorted, return_counts=True)
    Y = np.cumsum(counts)
    if density:
        Y = Y / len(X)

    # Plot the ECDF
    ax.vlines(X_unique[0], 0.0, Y[0], color=color, linestyle=linestyle)
    ax.step(X_unique, Y, where='post', color=color, linestyle=linestyle, label=label)

    # Optional bounds:
    if lb is not None:
        ax.hlines(0.0, lb, X_unique[0], color=color, linestyle=linestyle)
    if ub is not None:
        ax.hlines(1.0, X_unique[-1], ub, color=color, linestyle=linestyle)
        
