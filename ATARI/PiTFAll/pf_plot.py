import matplotlib.pyplot as plt



def violin(violin_data, violin_labels, violin_colors, fom):

    fig = plt.figure()

    ## violin plot
    bindat = plt.violinplot(violin_data,
                            showextrema=True, showmeans=True, vert=False,
                            widths=0.75)

    for i, pc in enumerate(bindat['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    bindat['cbars'].set(colors='k', linewidth=0.5, visible=True)
    bindat['cmins'].set(colors='k', label='min/max')
    bindat['cmaxes'].set(colors='k')
    bindat['cmeans'].set(colors='r', label='mean')

    ax = plt.gca()
    ax.set_yticks(range(1, len(violin_data)+1) )
    ax.set_yticklabels(violin_labels, rotation=45)
    ax.set_xlabel(fom)
    ax.legend()

    fig.tight_layout()

    return fig


# def plot(case_file, isample, fit_name):

#         # read sample_case data 
#         exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, fit_name)

#         # create plot
#         fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
#         # plot trans
#         ax[0].errorbar(exp_pw_df.E, exp_pw_df.exp_trans, yerr=np.sqrt(np.diag(exp_cov)), zorder=0, 
#                                 fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
#         ax[0].plot(exp_pw_df.E, exp_pw_df.theo_trans, lw=2, color='g', label='sol', zorder=1)
#         ax[0].plot(exp_pw_df.E, exp_pw_df[f'est_trans_{fit_name}'], lw=1, color='r', label='est', zorder=2)
#         ax[0].set_ylim([-.1, 1]); #ax[0].set_xscale('log')
#         ax[0].set_xlabel('Energy'); ax[0].set_ylabel('Transmission')
#         ax[0].legend()
#         # plot xs
#         ax[1].plot(theo_pw_df.E, theo_pw_df.theo_xs, lw=2, color='g', label='sol', zorder=1)
#         ax[1].plot(theo_pw_df.E, theo_pw_df[f'est_xs_{fit_name}'], lw=1, color='r', label='est', zorder=2)
#         ax[1].set_yscale('log'); #ax[1].set_xscale('log')
#         ax[1].set_xlabel('Energy'); ax[1].set_ylabel('Total Cross Section')
#         ax[1].legend()
#         fig.tight_layout()

#         return fig




#### hist and violin

# fig, (ax1,ax2) = subplots(1,2, figsize=(11,5))

# ## histogram
# bindat = ax1.hist(violin_data[0], bins=bins, density=True, alpha=0.5, label=violin_labels[0], color=violin_colors[0])
# bindat = ax1.hist(violin_data[1], bins=bins, density=True, alpha=0.5, label=violin_labels[1], color=violin_colors[1])
# bindat = ax1.hist(violin_data[2], bins=bins, density=True, alpha=0.5, label=violin_labels[2], color=violin_colors[2])
# ax1.legend()
# ax1.set_ylabel('Density')
# ax1.set_xlabel(fom)


# ## violin plot
# bindat = ax2.violinplot(violin_data,
#                     showextrema=True, showmeans=True, vert=False,
#                     widths=0.75)

# for i, pc in enumerate(bindat['bodies']):
#     pc.set_facecolor(violin_colors[i])
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.5)
    
# bindat['cbars'].set(colors='k', linewidth=0.5, visible=True)
# bindat['cmins'].set(colors='k', label='min/max')
# bindat['cmaxes'].set(colors='k')
# bindat['cmeans'].set(colors='r', label='mean')

# ax2.set_yticks([1, 2, 3])
# ax2.set_yticklabels(violin_labels, rotation=45)

# ax2.set_xlabel(fom)
# # ax2.set_ylabel('Fitting Method')
# ax2.legend()

# tight_layout()