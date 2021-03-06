import numpy as np
import os
import matplotlib.pyplot as plt

import mrestimator as mre
from help_functions import get_Fc, get_cell_nums, deconvolve_Fc, fit_tau, calc_signal

def compare_injected_vs_gen():
    paths = ['/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/2019-11-08_RL065_t-003/suite2p/plane0',
             '/data.nst/share/data/packer_calcium_mice/2019-03-01_R024/Spontaneous/suite2p/plane0',
             "/data.nst/share/data/packer_calcium_mice/2019-08-15_RL055_t-003",
             '/data.nst/share/data/packer_calcium_mice/2019-08-14_J059_t-002',
             '/data.nst/jdehning/packer_data/2019-11-07_J061_t-003/suite2p/plane0']
    fs_list = [30,30,30,30,15]
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
    for i_exp in range(len(paths)):
        snr_diff_2D_list.append([])
        for nth_largest in nth_largest_list:
            snr_periods = [None, None]
            for i_period in range(2):
                Fc_all = Fc_list[i_exp]
                dcnv_all = dcnv_list[i_exp]
                Fc = Fc_all[:, Fc_all.shape[1]//2:] if i_period == 0 else Fc_all[:, :Fc_all.shape[1]//2]
                dcnv = dcnv_all[:, dcnv_all.shape[1]//2:] if i_period == 0 else dcnv_all[:, :dcnv_all.shape[1]//2]
                snr = calc_signal(dcnv, n_bins_rolling_sum, nth_largest)/np.std(Fc, axis=-1)
                print(i_exp, nth_largest, snr[1])
                snr_periods[i_period] = np.array(snr)
            #if i_exp == 2:
            #    plt.plot(snr_periods[0], snr_periods[1], '.', alpha=0.4)
            #print(np.corrcoef(snr_periods[0], snr_periods[1]))
            #median_diff = np.median(np.abs((snr_periods[1]-snr_periods[0])/snr_periods[0]))
            median_diff =  np.corrcoef(snr_periods[0], snr_periods[1])[0,1]
            snr_diff_2D_list[-1].append(median_diff)
        snr_two_periods_list.append((snr_periods[0], snr_periods[1]))

    if False:
        f, axes = plt.subplots(1, 5, figsize = (15,3))
        titles = ['transgenic: 30 Hz, 30 min (2019-11-08, RL065)', 'transgenic: 30 Hz, 10 min (2019-03-01, RL024)',
                  'injected: 30 Hz, 26 min (2019-08-15, RL055)', 'injected: 30 Hz, 25 min (2019-08-14, J059)',
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


    snr_2Dlist = []

    range_snr = [[1,1.5],[1.5,2],[2,2.5],[2.5,3], [3,4]]
    coefficients_list = []

    for Fc_mat, act_mat, tau_mat, i_exp in zip(Fc_list, dcnv_list, tau_2Dlist, range(1000)):
        snr_2Dlist.append([])
        i_plot = 0
        coefficients_list.append([])
        for i_hist in range(len(range_snr)):
            coefficients_list[-1].append([])
        for Fc, act, tau in zip(Fc_mat, act_mat, tau_mat):
            #l_norm = 10
            #act_for_snr = act
            #act_for_snr = np.concatenate([np.sum([act[:-2], act[1:-1], act[2:]], axis=0), [0,0]])
            #act_for_snr = np.concatenate([np.sum([act[:-1], act[1:]], axis=0), [0]])
            snr = calc_signal(act, n_bins_rolling_sum, nth_largest_snr) / np.std(Fc)
            #snr = np.max(act_for_snr)/np.std(Fc)
            #snr = np.sum(act_for_snr**l_norm)**(1/l_norm)/np.std(Fc)
            if snr < 2 and snr > 1.5 and i_exp==0 and False:
                print(snr, tau)
                fs = fs_list[i_exp]
                #plt.clf()
                #f, axes = plt.subplots(2, 1, figsize=(10, 5))
                #axes[0].plot(Fc)
                #axes[0].plot(act_for_snr)
                #amax = np.argmax(act_for_snr)
                #axes[0].set_xlim(amax-fs*2, amax+fs*5)
                coefficients.append(mre.coefficients(act, k_arr, dt=1/fs* 1000, numboot=0, method='ts').coefficients)
                #axes[1].plot(mre.coefficients(act, k_arr, dt=1/fs* 1000, numboot=0, method='ts').coefficients)
                #input()
                plt.close()
            for i_hist, (min_snr, max_snr) in enumerate(range_snr):
                if snr > min_snr and snr < max_snr:
                    fs = fs_list[i_exp]
                    k_arr = np.arange(1, 2*fs)
                    coefficients_list[-1][i_hist].append(mre.coefficients(act, k_arr, dt=1/fs* 1000, numboot=0, method='ts').coefficients)
            snr_2Dlist[-1].append(snr)

    #plt.plot(np.mean(np.array(coefficients), axis=0))
    #plt.show()



    f, axes = plt.subplots(4, 5, figsize = (18,12))
    titles = ['transgenic: 30 Hz, 30 min\n(2019-11-08, RL065)', 'transgenic: 30 Hz, 10 min\n(2019-03-01, RL024)',
              'injected: 30 Hz, 26 min\n(2019-08-15, RL055)', 'injected: 30 Hz, 25 min\n(2019-08-14, J059)',
              'injected: 15 Hz, 20 min\n(2019-11-07, J061)']
    for i_ax, ax in enumerate(axes[0]):
        ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        #ax.hist(snr_2Dlist[i_ax], bins=np.linspace(0,8,30))
        ax.set_ylim(0,400)
        ax.set_xlabel("Signal-to-noise ratio")
        if i_ax == 0:
            ax.set_ylabel('timescales (ms)')
        ax.set_xlim(0,6)
        ax.set_title(titles[i_ax])
    for i_ax, ax in enumerate(axes[1]):
        #ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        ax.hist(snr_2Dlist[i_ax], bins=np.linspace(0,8,30))
        ax.set_ylim(0,200)
        ax.set_xlabel("Signal-to-noise ratio")
        if i_ax == 0:
            ax.set_ylabel('number of cells')
        ax.set_xlim(0,6)
    for i_ax, ax in enumerate(axes[2]):
        #ax.plot(snr_2Dlist[i_ax], tau_2Dlist[i_ax], '.', alpha=0.3)
        for i_hist, (min_snr, max_snr) in enumerate(range_snr):
            fs = fs_list[i_ax]
            k_arr = np.arange(1, 2 * fs)
            time_arr = k_arr/fs*1000
            correlation_coeff = np.mean(np.array(coefficients_list[i_ax][i_hist]), axis=0)
            num_cells = np.array(coefficients_list[i_ax][i_hist]).shape[0]
            try:
                ax.plot(time_arr, correlation_coeff,
                         label = 'cells with SNR between {} and {}\nnumber of cells: {}'.format(min_snr, max_snr, num_cells))
            except ValueError:
                continue
        ax.set_xlabel("Time difference [ms]")
        ax.legend(fontsize=7)
        if i_ax == 0:
            ax.set_ylabel('Average autocorrelation')
        #ax.set_xlim(0,10)
    for i_ax, ax in enumerate(axes[3]):
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
    plt.savefig('../reports/snr_of_recordings/snr_overview_figure.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    compare_injected_vs_gen()