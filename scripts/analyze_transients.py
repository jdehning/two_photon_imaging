import sys
import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage

import mrestimator as mre
from help_functions import get_Fc, get_cell_nums, deconvolve_Fc, fit_tau, calc_signal, calc_snr_mloidolt, calc_skewness_mloidolt, calc_brightness_mloidolt,\
    calc_snr_jdehning

from mpl_toolkits.mplot3d import Axes3D

def analyze_transients():
    paths = ['/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/2019-11-08_RL065_t-003/suite2p/plane0',
             '/data.nst/share/data/packer_calcium_mice/2019-03-01_R024/Spontaneous/suite2p/plane0',
             "/data.nst/share/data/packer_calcium_mice/2019-08-15_RL055_t-003",
             '/data.nst/share/data/packer_calcium_mice/2019-08-14_J059_t-002']
   #          '/data.nst/jdehning/packer_data/2019-11-07_J061_t-003/suite2p/plane0']
    fs_list = [30, 30, 30, 30, 30]
    tau_dcnv = 1.5
    Fc_list = [get_Fc(path)[get_cell_nums(path)] for path in paths]
    flu_vape, run = open_vape_fluerescence()
    Fc_list.append(flu_vape)
    mpl.rcdefaults()

    dcnv_list = [deconvolve_Fc(Fc, fs, tau=tau_dcnv) for Fc, fs in zip(Fc_list, fs_list)]
    tau_2Dlist = []
    for act_mat, fs in zip(dcnv_list, fs_list):
        tau_2Dlist.append([])
        for act in act_mat:
            tau = fit_tau(act, fs, k_arr=np.arange(1, 70))
            tau_2Dlist[-1].append(tau)
        tau_2Dlist[-1] = np.array(tau_2Dlist[-1])



    skew_2Dlist = []
    for Fc_mat, act_mat, tau_mat, i_exp in zip(Fc_list, dcnv_list, tau_2Dlist, range(1000)):
        skew_2Dlist.append([])
        for Fc, act, tau in zip(Fc_mat, act_mat, tau_mat):
            dcnv_m = act.reshape(1, act.shape[0])
            Fc_m = Fc.reshape(1, Fc.shape[0])

            snr = calc_skewness_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])[0]

            skew_2Dlist[-1].append(snr)
        skew_2Dlist[-1] = np.array(skew_2Dlist[-1])


    f, axes = plt.subplots(5, len(fs_list), figsize = (25,16))
    titles = ['transgenic: 30 Hz, 30 min\n(2019-11-08, RL065)',
             'transgenic: 30 Hz, 10 min\n(2019-03-01, RL024)',
              'injected: 30 Hz, 26 min\n(2019-08-15, RL055)', 'injected: 30 Hz, 25 min\n(2019-08-14, J059)',
            #  'injected: 15 Hz, 20 min\n(2019-11-07, J061)',
              'injected: 30 Hz, 73 min\n(2019-12-13, J064)']
    for i_exp in range(len(titles)):
        axes_exp = [ax[i_exp] for ax in axes]
        indices_small_tau = get_indices_largest_skew('small', skew_2Dlist[i_exp], tau_2Dlist[i_exp])
        if i_exp in [0,1]:
            indices_small_tau = np.random.choice(len(skew_2Dlist[i_exp]), replace=False, size=100)
            label_small = 'randomly chosen'
        else:
            label_small = 'small timescale chosen'
        indices_large_tau = get_indices_largest_skew('large', skew_2Dlist[i_exp], tau_2Dlist[i_exp])
        n_cells = 6

        axes_exp[0].plot(skew_2Dlist[i_exp], tau_2Dlist[i_exp], '.', alpha=0.3)
        axes_exp[0].plot(skew_2Dlist[i_exp][indices_small_tau[:n_cells]],tau_2Dlist[i_exp][indices_small_tau[:n_cells]], '.',
                         markersize=10, color='tab:orange', label=label_small)
        axes_exp[0].plot(skew_2Dlist[i_exp][indices_large_tau[:n_cells]],tau_2Dlist[i_exp][indices_large_tau[:n_cells]],'.',
                         markersize=10, color='tab:pink', label='large timescale chosen')
        #ax.hist(snr_2Dlist[i_ax], bins=np.linspace(0,8,30))
        axes_exp[0].set_ylim(0,400)
        axes_exp[0].set_xlabel("skewness")
        if i_exp == 0:
            axes_exp[0].set_ylabel('timescales (ms)')
        axes_exp[0].set_title(titles[i_exp])
        axes_exp[0].legend()


        time_to_plot = 60*3
        len_to_plot = time_to_plot*fs_list[i_exp]
        x = np.arange(len_to_plot)/fs_list[i_exp]
        for i in range(n_cells):
            offset = np.std(Fc_list[i_exp])*0.5
            smooth = lambda x: scipy.ndimage.gaussian_filter1d(x, 6)
            axes_exp[1].plot(x, smooth(Fc_list[i_exp][indices_small_tau[i], :len_to_plot]) + i*offset, color='tab:orange', alpha=0.7)
            axes_exp[2].plot(x, smooth(Fc_list[i_exp][indices_large_tau[i], :len_to_plot]) + i*offset, color='tab:pink', alpha=0.7)
        if i_exp == 0:
            axes_exp[1].set_ylabel('fluorescence unnormed\nsmall timescale\nsmoothed over 6 frames')
            axes_exp[2].set_ylabel('fluorescence unnormed\nlarge timescale\nsmoothed over 6 frames')
        axes_exp[1].set_xlabel("time (s)")
        axes_exp[2].set_xlabel("time (s)")

        for i in range(n_cells):
            plot_max_transient(indices_small_tau[i], dcnv_list[i_exp], Fc_list[i_exp], axes_exp[3], color='tab:orange', alpha=0.7)
            plot_max_transient(indices_large_tau[i], dcnv_list[i_exp], Fc_list[i_exp], axes_exp[4], color='tab:pink', alpha=0.7)

        if i_exp == 0:
            axes_exp[3].set_ylabel('maximal fluorescent transient normed\nsmall timescale\nsmoothed over 3 frames')
            axes_exp[4].set_ylabel('maximal fluorescent transient normed\nlarge timescale\nsmoothed over 3 frames')
        axes_exp[3].set_xlabel("time from burst (s)")
        axes_exp[4].set_xlabel("time from burst (s)")

    plt.tight_layout()
    plt.savefig('../reports/analyze_transients/large_figure_transients.pdf')
    plt.show()

def analyze_specific_experiment():
    Fc, run = open_vape_fluerescence()
    tau_dcnv = 1.5
    fs = 30
    n_cells = 6
    dcnv = deconvolve_Fc(Fc, fs, tau=tau_dcnv)
    ops1 = np.load('/data.nst/share/data/packer_calcium_mice/jimmy/J064/run10_ops1.npy', allow_pickle=True)

    skew_arr = []
    for Fc_arr, act in zip(Fc, dcnv):
        dcnv_m = act.reshape(1, act.shape[0])
        Fc_m = Fc_arr.reshape(1, Fc_arr.shape[0])

        snr = calc_skewness_mloidolt(dcnv_m, Fc_m, fs)[0]
        skew_arr.append(snr)
    skew_arr = np.array(skew_arr)

    tau_arr = []
    for act in dcnv:
        tau = fit_tau(act, fs, k_arr=np.arange(1, 70))
        tau_arr.append(tau)
    tau_arr = np.array(tau_arr)

    indices_small_tau = get_indices_largest_skew('small', skew_arr, tau_arr)
    indices_large_tau = get_indices_largest_skew('large', skew_arr, tau_arr)


    f, axes = plt.subplots(5, figsize=(12, 30))

    im = np.zeros((514, 1024))
    for i in indices_small_tau[:n_cells]:
        im[run.stat[i]['ypix'], run.stat[i]['xpix']] =  run.stat[i]['lam']
        plot_circle_around_ROI(i, 'tab:orange', axes[0], run)
    for i in indices_large_tau[:n_cells]:
        im[run.stat[i]['ypix'], run.stat[i]['xpix']] = run.stat[i]['lam']
        plot_circle_around_ROI(i, 'tab:pink', axes[0], run)
    axes[0].set_title('ROIs mask')

    axes[0].imshow(im)

    axes[1].imshow(ops1[0]['meanImg'])
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[1], run)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[1], run)
    axes[1].set_title('mean image')


    axes[2].imshow(ops1[0]['meanImgE'])
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[2], run)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[2], run)
    axes[2].set_title('mean image enhanced')

    axes[3].imshow(ops1[0]['max_proj'])
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[3], run, -3, -3)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[3], run, -3, -3)
    axes[3].set_title('maximal projection')
    plt.tight_layout()

    axes[4].imshow(ops1[0]['Vcorr'])
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[4], run, -3, -3)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[4], run, -3, -3)
    axes[4].set_title('Correlation map')
    plt.tight_layout()
    plt.savefig('../reports/analyze_transients/weird_and_normal_cell_in_2019-12-13_J064.pdf')

def plot_circle_around_ROI(index, color, ax, run, add_x = 0, add_y = 0):
    y = np.mean(run.stat[index]['ypix'])+ add_y
    x = np.mean(run.stat[index]['xpix']) + add_x
    circle = plt.Circle((x, y), 15, edgecolor=color, facecolor=(1, 1, 0, 0),
                        linewidth=3)
    ax.add_patch(circle)

def get_indices_largest_skew(large_or_small_tau, skew_arr, tau_arr):
    if large_or_small_tau == 'large':
        skew_arr_for_large = np.copy(skew_arr)
        skew_arr_for_large[~((tau_arr > 50) & (tau_arr < 400))] = 0
        indices_sorted_large_tau = np.argsort(skew_arr_for_large)[::-1]
        return indices_sorted_large_tau
    elif large_or_small_tau == 'small':
        skew_arr_for_small = np.copy(skew_arr)
        skew_arr_for_small[~(tau_arr < 20)] = 0
        indices_sorted_small_tau = np.argsort(skew_arr_for_small)[::-1]
        return indices_sorted_small_tau

def plot_max_transient(i_cell, dcvn, Fc, ax, alpha=0.3, color='tab:blue'):
    dcvn_smoothed = scipy.ndimage.gaussian_filter1d(dcvn[i_cell], 10)*10
    max_i = np.argmax(dcvn_smoothed)
    Fc_max = scipy.ndimage.gaussian_filter1d(Fc[i_cell], 5)[max_i + 20]
    time_to_plot = 25
    fs = 30
    len_to_plot = time_to_plot * fs
    x = np.arange(len_to_plot) / fs - 5
    if max_i > len(dcvn_smoothed) - len_to_plot:
        return

    transient = Fc[i_cell][max_i - 5*fs:max_i + 20*fs] / Fc_max
    transient = scipy.ndimage.gaussian_filter1d(transient, 3)
    ax.plot(x, transient, color=color, alpha=alpha)

def plot_n_max_transients(i_cell, dcvn, Fc, n_transients, alpha=0.3, color='tab:blue'):
    dcnv_tmp = np.copy(dcvn[i_cell])
    for i in range(n_transients):
        dcvn_smoothed = scipy.ndimage.gaussian_filter1d(dcnv_tmp, 20) * 20
        max_i = np.argmax(dcvn_smoothed)
        if max_i > len(dcvn_smoothed) -1000:
            dcnv_tmp[max_i] -= dcvn_smoothed[max_i]
            continue
        Fc_max = scipy.ndimage.gaussian_filter1d(Fc[i_cell],5)[max_i+20]
        transient = Fc[i_cell][max_i - 300:max_i + 1000] / Fc_max
        transient = scipy.ndimage.gaussian_filter1d(transient, 10)
        plt.plot(transient, color=color, alpha=alpha)
        dcnv_tmp[max_i] -= dcvn_smoothed[max_i]


def open_vape_fluerescence():
    path_to_vape = os.path.expanduser('/home/jdehning/ownCloud/studium/two_photon_vape')
    sys.path.append(path_to_vape)
    sys.path.append(os.path.join(path_to_vape, 'utils'))
    import utils.utils_funcs as utils

    # from subsets_analysis import Subsets
    import pickle

    # %% md

    ## Loading the Data

    # %%

    # dictionary of mice and run numbers to analyse
    run_dict = {
        'J064': [10]
    }

    # %%

    # local path to behaviour pickle files
    # this takes a while to load so maybe should do some further caching in the future
    pkl_path = os.path.expanduser('/data.nst/share/data/packer_calcium_mice/jimmy')

    runs = []
    for mouse in run_dict:
        for run_number in run_dict[mouse]:
            run_path = os.path.join(pkl_path, mouse, 'run{}.pkl'.format(run_number))
            with open(run_path, 'rb') as f:
                r = pickle.load(f)
                runs.append(r)

    # %%

    run = runs[0]  # just take the first session in the dict for now
    run.__dict__.keys()

    # %% md

    ## Basic attributes of the run object

    # %%

    # toggle this cell on if you want to use raw fluoresence instead of processed (neuropil subtracted and Df/f)
    # for i in range(len(runs)):
    #     runs[i].flu = runs[i].flu_raw

    # %%

    # processed (neuropil subtracted and Df/f) fluoresence matrix from first run
    flu = run.flu
    return flu, run
