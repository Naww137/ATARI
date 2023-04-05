#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:16:02 2022

@author: noahwalton
"""

from matplotlib.pyplot import *
import numpy as np



def plot1(energy,theo,exp,label1,label2):
    
    plot(energy, theo, label=label1, zorder=2)
    scatter(energy,exp, label=label2, s=1, c='k', zorder=1)
    
    legend()
    #plt.yscale('log'); 
    xscale('log')
    show();close()
    

def plot2(x,theo,exp,exp_unc, title):
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True, constrained_layout=True, gridspec_kw={'height_ratios': [2, 1, 1]}) # , figsize=(12,5)
    rcParams['figure.dpi'] = 500
    
    ax1.plot(x,theo, lw=0.5, color='b', label='$T_{theo}$', zorder=2)
    #ax1.scatter(energy,exp, s=0.1, c='r', label='$T_{exp}$')
    ax1.errorbar(x, exp, yerr=exp_unc, color='k',ecolor='k',elinewidth=1,capsize=2, fmt='.', ms=3, label='$T_{exp}$', zorder=0)
    
    ax1.legend()
    ax1.set_ylabel('T') #('$\sigma$')
    #ax1.set_yscale('log'); 
    #ax1.set_xscale('log')
    ax1.set_ylim([0,max(exp)+0.1])
    
    rel_se = np.sqrt((exp-theo)**2) #/theo
    ax2.scatter(x, rel_se, s=2)
    #ax2.set_ylim([-.5,.5])
    ax2.set_ylabel('L2 Norm'); #ax2.set_ylabel('L1 Norm (relative)')
    
    ax3.scatter(x, exp_unc, lw=0.5, color='b', s=2, zorder=2)
    ax3.set_ylabel('$\delta$T') #('$\sigma$')
    ax3.set_xlabel('ToF (s)');
    
    suptitle(title)
    tight_layout()
    show(); close()
    
    
def exp_theo(tof, Tn, dT, T_theo):
    figure()
    errorbar(tof,Tn, yerr=dT,color='r',ecolor='k',elinewidth=1,capsize=2, fmt='.', ms=3)
    #scatter(tof, Tn, label='Experimental', s=1, c='k')
    plot(tof, T_theo, label='Theoretical', c='g', lw=0.25)
    
    legend()
    ylim([-1,5])
    #xlim([1e2,1e3])
    xscale('log')
    #yscale('log')
    

def unc_noise_theo(tof, theo, exp, unc):

    fig, (ax1, ax2, ax3) = subplots(3,2, gridspec_kw={'height_ratios': [1, 1, 1]}, sharex=True, figsize=(15,6)) # , figsize=(12,5)

    ax1[0].scatter(tof, unc/exp*100, lw=0.5, color='b', s=0.5, zorder=2)
    # ax1[0].set_ylim([0,100]);
    ax1[0].set_xlim([1e2,2e3])
    ax1[0].set_yscale('log')
    ax1[0].set_ylabel('$\delta$T/T'); #('$\sigma$')

    ax2[0].plot(tof, theo, lw= 0.5, c='orange')
    #ax2[0].set_yscale('log')
    ax2[0].set_xlim([1e2,2e3])
    ax2[0].set_ylabel('Trans')
    ax2[0].legend()

    rel_se = (exp-theo)/theo
    ax3[0].scatter(tof, rel_se, s=0.5, c='b')
    ax3[0].set_ylim([0.001,100])
    ax3[0].set_ylabel('Rel Noise')
    ax3[0].set_yscale('log')


    # cross section
    title('Cross Section')
    ax1[1].scatter(tof, unc/exp*100, lw=0.5, color='b', s=0.5, zorder=2)
    # ax1[1].set_ylim([3,100]);
    ax1[1].set_xlim([1e2,2e3])
    ax1[1].set_yscale('log')
    ax1[1].set_ylabel('$\delta\sigma/\sigma$'); #('$\sigma$')

    ax2[1].plot(tof, theo, lw= 0.5, c='orange')
    ax2[1].set_yscale('log')
    ax2[1].set_xlim([1e2,2e3])
    ax2[1].set_ylabel(r'$\sigma_{t}$')
    ax2[1].legend()

    rel_se = (exp.trans.exp_xs-exp.trans.theo_xs)/exp.trans.theo_xs
    ax3[1].scatter(tof, rel_se, s=0.5, c='b')
    # ax3[1].set_ylim([0.001,1000])
    ax3[1].set_ylabel('Rel Noise')
    ax3[1].set_yscale('log')


    xscale('log')
    xlabel('ToF (s)');
    suptitle('Uncertainty and Noise on Transmission')
    tight_layout()
    #plt.show(); plt.close()

    