import numpy as np
from numpy import ndarray
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt

__doc__ = """
This module is used to analyze the data created by TAZ with scores, confusion matrices, probability
distributions, etc.
"""

def _fraction_estimation(N:int, n:int):
    """
    A function used to estimate the fraction of successful counts when restricted between 0 and 1.
    The estimated fraction and standard deviation are derived from the expectation value and
    variance of the binomial probability distribution.

    Parameters
    ----------
    N : int
        The total number of trials.
    n : int
        The number of successful results from the `N` trials.

    Returns
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

def spingroup_names(num_groups:int, last_false:bool=True):
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
    if num_groups > 23:
        raise ValueError(f'You are implying you have {num_groups} groups. I do not have enough letters in the alphabet to differentiate each spingroup!')
    
    # Assigning spingroup names:
    letters = 'ABCDEGHJKLMNPQRSTUVWXYZ' # some letters removed to avoid confusion.
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
    elif metric == 'brier score':
        score = np.mean(np.sum(pred_probs**2, axis=1)
                        + (1-2*pred_probs[:,answer]))
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
    elif metric == 'Brier score':
        score = np.mean(np.sum(pred_probs*(1-pred_probs), axis=1))
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
        sg_names = spingroup_names(num_groups)
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

def correlate_probabilities(pred_probs:ndarray, answer:ndarray):
    """
    ...

    Parameters
    ----------
    pred_probs : array[float]
        Predicted spingroup probabilties for each resonance.
    answer : array[int]
        The number of correct solutions.

    Returns
    -------
    Qs : list[array[float]]
        The pmf values for the frequency of each probability bin for each spingroup.

    See Also:
    --------
    ProbCorrPlot
    """

    num_groups = pred_probs.shape[1]

    nBin = round(np.sqrt(len(answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    Qs = []
    for g in range(num_groups):
        prob_guess_type = pred_probs[:,g]
        prob_guess_cor  = prob_guess_type[answer == g]
        count_all = np.histogram(prob_guess_type, bins=edges)[0]
        count_cor = np.histogram(prob_guess_cor , bins=edges)[0]

        # Only plot the cases where counts exist:
        count_all_non_zero = count_all[count_all > 1]
        count_cor_non_zero = count_cor[count_all > 1]

        # Find the expected mean probability:
        prob_expected_non_zero = []
        for binID in range(nBin):
            if count_all[binID] <= 1:
                continue
            prob_expected_non_zero.append(np.mean(prob_guess_type[(prob_guess_type > edges[binID]) & (prob_guess_type < edges[binID+1])]))
        prob_expected_non_zero = np.array(prob_expected_non_zero)

        # Estimating probability and confidence:
        Q = np.zeros((len(count_all_non_zero),))
        for i, (count_all_, count_cor_, prob_expected_) in enumerate(zip(count_all_non_zero, count_cor_non_zero, prob_expected_non_zero)):
            Q[i] = binom(count_all_, prob_expected_).pmf(count_cor_) * count_all_
        Qs.append(Q)
    return Qs

def ProbCorrPlot(pred_probs:ndarray, answer:ndarray,
                 sg_names:list=None, image_name:str=None, fig_num:int=None):
    """
    Groups resonances into bins based on their predicted probabilities and plots the frequency
    of correct assignments versus the binned probabilities.

    Parameters
    ----------
    pred_probs : array[float]
        Predicted spingroup probabilties for each resonance.
    answer     : array[int]
        The number of correct solutions.
    sg_names   : list, optional
        A list of names for each spingroup. Default is "A", "B", "C", etc.
    image_name : str, optional
        The name of the image to save with a format specifier, "sgn" for the spingroup name. If
        fig_name is not provided, plt.show is used instead of saving the image.
    fig_num    : int, optional
        The figure number.

    See Also:
    --------
    correlate_probabilities
    """

    num_groups = pred_probs.shape[1]
    if sg_names is None:
        sg_names = spingroup_names(num_groups)

    nBin = round(np.sqrt(len(answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    for g in range(num_groups):
        prob_guess_all = pred_probs[:,g]
        prob_guess_cor = prob_guess_all[answer == g]
        count_all = np.histogram(prob_guess_all, bins=edges)[0]
        count_cor = np.histogram(prob_guess_cor, bins=edges)[0]

        # Only plot the cases where counts exist:
        count_all_non_zero = count_all[count_all > 1]
        count_cor_non_zero = count_cor[count_all > 1]

        # Find the expected mean probability:
        prob_expected = 0.5 * (edges[1:] + edges[:-1])
        prob_expected_non_zero = np.histogram(prob_guess_all, weights=prob_guess_all, bins=edges)[0][count_all > 1] / count_all[count_all > 1]
        prob_expected[count_all > 1] = prob_expected_non_zero

        # Estimating probability and confidence:
        prob_ans_cor_est, prob_ans_cor_std = _fraction_estimation(count_all_non_zero, count_cor_non_zero)

        # Plotting:
        plt.figure(fig_num+g)
        plt.clf()
        plt.errorbar(prob_expected_non_zero, prob_ans_cor_est, prob_ans_cor_std, capsize=3, ls='none', c='k')
        plt.scatter(prob_expected_non_zero, prob_ans_cor_est, marker='.', s=14, c='k')
        plt.plot([0,1], [0,1], ':b', lw=2, label='Ideal Correlation')

        # Probability density on the guessed probability:
        norm = 0.25/np.max(count_all)
        plt.fill_between(prob_expected, norm*count_all, alpha=0.8, color=(0.1,0.5,0.1), label='Probability Density')
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title(f'Probability Accuracy for Spin Group {sg_names[g]}', fontsize=18)
        plt.xlabel('Predicted Probability', fontsize=15)
        plt.ylabel('Fraction of Correct Assignments', fontsize=15)
        plt.legend(fontsize=10)

        if image_name is not None:
            image_name = str(image_name).format(sgn=sg_names[g])
            plt.savefig(f'{image_name}.png')
        else:
            plt.show()

def ecdf(X, lb:float=None, ub:float=None,
         color:str='k', linestyle:str='-',
         density:bool=True, label:str=None, ax=None):
    """
    Plots the empirical cumulative distribution function (ECDF) of a float array, `X`.

    Parameters
    ----------
    X         : float, array-like
        Values to plot the emiprical CDF for.
    lb        : float
        Optional lower bound.
    ub        : float
        Optional upper bound.
    color     : str
        The line color.
    linestyle : str
        The line style.
    density   : bool
        Determines whether cumulative densities are plotted (highest value is 1.0), or cumulative
        counts are plotted (highest value is the total number of values). Default is True.
    label     : str
        The legend label for the line.
    ax        : plt.axes
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