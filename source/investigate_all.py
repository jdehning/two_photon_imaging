import numpy as np
import os
import matplotlib.pyplot as plt

import mrestimator as mre
from help_functions import get_Fc, get_cell_nums, deconvolve_Fc, fit_tau, calc_signal, calc_snr_mloidolt, calc_skewness_mloidolt, calc_brightness_mloidolt,\
    calc_snr_jdehning

from mpl_toolkits.mplot3d import Axes3D

def check_condition(condition, snr, skew, bright):
    truth = True
    
    if snr < condition[0]:
        truth = False
    
    if skew < condition[1]:
        truth = False
    
    if bright > condition[2]:
        truth = False

    return truth

def compare_injected_vs_gen():
    paths = ['/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/2019-11-08_RL065_t-003/suite2p/plane0',
             '/data.nst/share/data/packer_calcium_mice/2019-03-01_R024/Spontaneous/suite2p/plane0',
             "/data.nst/share/data/packer_calcium_mice/2019-08-15_RL055_t-003",
             '/data.nst/share/data/packer_calcium_mice/2019-08-14_J059_t-002',
             '/data.nst/jdehning/packer_data/2019-11-07_J061_t-003/suite2p/plane0']
    fs_list = [30,30, 30,30,15]
    tau_dcnv = 1.5
    Fc_list = [get_Fc(path)[get_cell_nums(path)] for path in paths]
    dcnv_list = [deconvolve_Fc(Fc, fs, tau=tau_dcnv) for Fc, fs in zip(Fc_list, fs_list)]
    tau_2Dlist = []
    for act_mat, fs in zip(dcnv_list, fs_list):
        tau_2Dlist.append([])
        for act in act_mat:
            tau = fit_tau(act, fs, k_arr=np.arange(1,70))
            tau_2Dlist[-1].append(tau)
    nth_largest_snr = 5
    n_bins_rolling_sum = 3

    ##Calculate SNR with Jonas' method
    #nth_largest_list=np.round(np.logspace(np.log10(1),np.log10(100),20)).astype('int')
    nth_largest_list = [nth_largest_snr]
    snr_diff_2D_list = []
    snr_two_periods_list = []
    s
    for i_exp in range(len(paths)):
        snr_diff_2D_list.append([])
        for nth_largest in nth_largest_list:
            snr_periods = [None, None]
            for i_period in range(2):
                Fc_all = Fc_list[i_exp]
                dcnv_all = dcnv_list[i_exp]
                Fc = Fc_all[:, Fc_all.shape[1]//2:] if i_period == 0 else Fc_all[:, :Fc_all.shape[1]//2]
                dcnv = dcnv_all[:, dcnv_all.shape[1]//2:] if i_period == 0 else dcnv_all[:, :dcnv_all.shape[1]//2]
                snr = calc_brightness_mloidolt(dcnv, Fc, fs_list[i_exp])
                snr_periods[i_period] = np.array(snr)
                
                #snr = calc_signal(dcnv, n_bins_rolling_sum, nth_largest)/np.std(Fc, axis=-1)
                #print(i_exp, nth_largest, snr[1])
                #snr_periods[i_period] = np.array(snr)
            #if i_exp == 2:
            #    plt.plot(snr_periods[0], snr_periods[1], '.', alpha=0.4)
            #print(np.corrcoef(snr_periods[0], snr_periods[1]))
            #median_diff = np.median(np.abs((snr_periods[1]-snr_periods[0])/snr_periods[0]))
            median_diff =  np.corrcoef(snr_periods[0], snr_periods[1])[0,1]
            snr_diff_2D_list[-1].append(median_diff)
        snr_two_periods_list.append((snr_periods[0], snr_periods[1]))

    if False:
        f, axes = plt.subplots(1, 3, figsize = (15,3))
        titles = [#'transgenic: 30 Hz, 30 min (2019-11-08, RL065)', 
                  #'transgenic: 30 Hz, 10 min (2019-03-01, RL024)',
                  'injected: 30 Hz, 26 min (2019-08-15, RL055)', 
                  'injected: 30 Hz, 25 min (2019-08-14, J059)',
                  'injected: 15 Hz, 30 min (2019-11-07, J061)']
        for i_ax, ax in enumerate(axes):
            ax.plot(nth_largest_list, snr_diff_2D_list[i_ax])
            #ax.set_ylim(0,400)
            ax.set_xlabel("snr")
            if i_ax == 0:
                ax.set_ylabel('median percentual difference of snr\nbetween begin and end of recording')
            #ax.set_xlim(0,12)
            ax.set_title(titles[i_ax])
        plt.show()


    snr_2Dlist_mloidolt = []
    snr_2Dlist_jdehning = []
    bright_2Dlist = []
    skew_2Dlist = []

    conditions_list = [[2,1.5,300]]
    coefficients_list = []

    # calculate observables
    for Fc_mat, act_mat, tau_mat, i_exp in zip(Fc_list, dcnv_list, tau_2Dlist, range(1000)):
        coefficients_list.append([])

        snr_2Dlist_mloidolt.append([])
        snr_2Dlist_jdehning.append([])

        skew_2Dlist.append([])
        bright_2Dlist.append([])

        print(conditions_list)
        print(len(conditions_list))

        for i_hist in range(len(conditions_list)):
                coefficients_list[-1].append([])
        print(coefficients_list)

        i_plot = 0
        for Fc, act, tau in zip(Fc_mat, act_mat, tau_mat):
            dcnv_m = act.reshape(1, act.shape[0])
            Fc_m = Fc.reshape(1, Fc.shape[0])
            
            snr = calc_snr_jdehning(dcnv_m, Fc_m, fs_list[i_exp])
            snr_2Dlist_jdehning[-1].append(snr)

            snr = calc_snr_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])
            snr_2Dlist_mloidolt[-1].append(snr)

            
            bright = calc_brightness_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])
            bright_2Dlist[-1].append(bright)
            
            skew = calc_skewness_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])
            skew_2Dlist[-1].append(skew)

            for i_hist, condition in enumerate(conditions_list):
                truth = check_condition(condition, snr, skew, bright) 
                if truth:
                    fs = fs_list[i_exp]
                    k_arr = np.arange(1, 2*fs)
                    coefficients_list[-1][i_hist].append(mre.coefficients(act, 
                        k_arr, dt=1/fs* 1000, numboot=0, method='ts').coefficients)

    #plt.plot(np.mean(np.array(coefficients), axis=0))
    #plt.show()

    axes = []
    fig = plt.figure(figsize=(18,12))
    #f, axes = plt.subplots(4, 3, figsize = (18,12), projection='3d')
    titles = [#'transgenic: 30 Hz, 30 min\n(2019-11-08, RL065)', 
              #'transgenic: 30 Hz, 10 min\n(2019-03-01, RL024)',
              'injected: 30 Hz, 26 min\n(2019-08-15, RL055)', 
              'injected: 30 Hz, 25 min\n(2019-08-14, J059)',
              'injected: 15 Hz, 20 min\n(2019-11-07, J061)']
    
    for i_ax in range(len(titles)):
        #ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        #ax.hist(snr_2Dlist[i_ax], bins=np.linspace(0,8,30))
        #ax.set_ylim(0,400)
        ax = fig.add_subplot(4, 3, 1+i_ax, projection='3d')

        scat = ax.scatter(np.asarray(snr_2Dlist[i_ax]), np.asarray(skew_2Dlist[i_ax]), np.asarray(bright_2Dlist[i_ax]), 
                s=40, c=np.asarray(tau_2Dlist[i_ax]), cmap='magma', alpha=0.3)

        ax.set_zlabel("Brightness")
        ax.set_ylabel("Skewness")
        ax.set_xlabel("Signal-to-noise ratio")
        
        if i_ax == 0:
            ax.set_ylabel('timescales (ms)')
        fig.colorbar(scat)

        ax.set_title(titles[i_ax])

    for i_ax in range(len(titles)):
        #ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        ax = fig.add_subplot(4, 3, 4+i_ax)
        
        ax.hist(np.asarray(snr_2Dlist[i_ax]).flatten())
        median_snr = np.median(np.asarray(snr_2Dlist[i_ax]).flatten())
        ax.axvline(median_snr, label='Median'+str(median_snr))
        #ax.set_ylim(0,200)
        ax.legend(fontsize=7)
        ax.set_xlabel("Signal-to-noise ratio")
        if i_ax == 0:
            ax.set_ylabel('number of cells')
        #ax.set_xlim(0,30)

    for i_ax in range(len(titles)):
        ax = fig.add_subplot(4, 3, 7+i_ax)

        #ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        for i_hist, condition in enumerate(conditions_list):
            fs = fs_list[i_ax]
            k_arr = np.arange(1, 2 * fs)
            time_arr = k_arr/fs*1000
            #correlation_coeff = np.mean(np.array(coefficients_list[i_ax][i_hist]), axis=0)
            correlation_coeffs = np.array(coefficients_list[i_ax][i_hist])
            num_cells = np.array(coefficients_list[i_ax][i_hist]).shape[0]
            for i_cells in range(num_cells):
                if tau_2Dlist[i_ax][i_cells] < 10:
                    ax.plot(time_arr, correlation_coeffs[i_cells], label=str(tau_2Dlist[i_ax][i_cells]))
            #try:
            #    ax.plot(time_arr, correlation_coeffs,
            #             label = 'cells fulfilling [{},{},{}] of cells: {}'.format(*condition, num_cells))
            #except ValueError:
            #    continue
        ax.set_xlabel("Time difference [ms]")
        ax.legend(fontsize=7)
        if i_ax == 0:
            ax.set_ylabel('autocorrelation')
        #ax.set_xlim(0,10)

    for i_ax in range(len(titles)):
        ax = fig.add_subplot(4, 3, 10+i_ax)

        snr1 = snr_two_periods_list[i_ax][0]
        snr2 = snr_two_periods_list[i_ax][1]
        ax.plot(snr1,snr2, '.', alpha =0.4)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlim(ax.get_ylim())
        ax.set_xlabel("SNR first half of recording\nCorrelation coefficient: {:.2f}".format(np.corrcoef(snr1,snr2)[0,1]))
        ax.plot([0,ax.get_xlim()[1]], [0,ax.get_ylim()[1]], ':' ,color='tab:gray')
        if i_ax == 0:
            ax.set_ylabel("SNR second half of recording")
    plt.tight_layout()
    plt.savefig('../reports/snr_of_recordings/all_mloidolt.pdf')
    plt.show()

if __name__ == '__main__':
    compare_injected_vs_gen()
