import sys
import os

sys.path.append('../source')
sys.path.append('/home/loidolt/Projects/Vape/my_suite2p')


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage

import mrestimator as mre
from help_functions import get_Fc, get_cell_nums, deconvolve_Fc, fit_tau, calc_signal, calc_snr_mloidolt, calc_skewness_mloidolt, calc_brightness_mloidolt,\
    calc_snr_jdehning

from mpl_toolkits.mplot3d import Axes3D

# shut up mr. estimator
mre.ut._logstreamhandler.setLevel('ERROR')

def gate_newData_tauSelect():
    paths = ["/data.nst/share/data/rob_calcium_mice/2019-11-14_J061",
             "/data.nst/share/data/rob_calcium_mice/2019-11-14_J063",
             "/data.nst/share/data/rob_calcium_mice/2019-11-15_J064",
             "/data.nst/share/data/rob_calcium_mice/2019-11-18_J065"]
 
    #          '/data.nst/jdehning/packer_data/2019-11-07_J061_t-003/suite2p/plane0']
    tau_dcnv = 1.5
    #Fc_list = [get_Fc(path)[get_cell_nums(path)] for path in paths]
    flu_vape, run, fs_list, whisker_stim_flu_vape = open_vape_fluerescence_mloidolt()
    Fc_list = flu_vape
    mpl.rcdefaults()

    dcnv_list = [deconvolve_Fc(Fc, fs, tau=tau_dcnv) for Fc, fs in zip(Fc_list, fs_list)]
    dcnv_whisker_list = [deconvolve_Fc(Fc, fs, tau=tau_dcnv) for Fc, fs in zip(whisker_stim_flu_vape, fs_list)]

    tau_2Dlist = []
    whisker_tau_2Dlist = []
    raw_tau_2Dlist = []
    
    tau_firstHalf_2Dlist = []
    tau_secondHalf_2Dlist = []

    for act_mat, fs, raw_mat, whisker_mat in zip(dcnv_list, fs_list, Fc_list, dcnv_whisker_list):
       tau_2Dlist.append([])
       whisker_tau_2Dlist.append([])
       raw_tau_2Dlist.append([])

       tau_firstHalf_2Dlist.append([])
       tau_secondHalf_2Dlist.append([])

       for act, raw, whisker_act in zip(act_mat, raw_mat, whisker_mat):
           tau = fit_tau(act, fs, k_arr=np.arange(1, 70))
           tau_2Dlist[-1].append(tau)
           
           tau_len_half = int(np.floor(len(act)/2))

           tau_firstHalf = fit_tau(act[:tau_len_half], fs, k_arr=np.arange(1, 70))
           tau_firstHalf_2Dlist[-1].append(tau_firstHalf)

           tau_secondHalf = fit_tau(act[tau_len_half:], fs, k_arr=np.arange(1, 70))
           tau_secondHalf_2Dlist[-1].append(tau_secondHalf)
          
           #raw_tau = fit_tau(raw, fs, k_arr=np.arange(1, 70))
           #raw_tau_2Dlist[-1].append(raw_tau)

           whisker_tau = fit_tau(whisker_act, fs, k_arr=np.arange(1, 70))
           whisker_tau_2Dlist[-1].append(whisker_tau)


       tau_firstHalf_2Dlist[-1] = np.array(tau_firstHalf_2Dlist[-1])
       tau_secondHalf_2Dlist[-1] = np.array(tau_secondHalf_2Dlist[-1])

       tau_2Dlist[-1] = np.array(tau_2Dlist[-1])
       raw_tau_2Dlist[-1] = np.array(raw_tau_2Dlist[-1])
       whisker_tau_2Dlist[-1] = np.array(whisker_tau_2Dlist[-1])

    coefficients_2Dlist = []
    raw_coefficients_2Dlist = []
    skew_2Dlist = []
    bright_2Dlist = []
    for Fc_mat, act_mat, tau_mat, i_exp in zip(Fc_list, dcnv_list, tau_2Dlist, range(1000)):
        skew_2Dlist.append([])
        bright_2Dlist.append([])
        coefficients_2Dlist.append([])
        raw_coefficients_2Dlist.append([])
        for Fc, act, tau in zip(Fc_mat, act_mat, tau_mat):
            dcnv_m = act.reshape(1, act.shape[0])
            Fc_m = Fc.reshape(1, Fc.shape[0])

            #bright = calc_brightness_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])[0]
            #skew = calc_skewness_mloidolt(dcnv_m, Fc_m, fs_list[i_exp])[0]

            #bright_2Dlist[-1].append(bright)
            #skew_2Dlist[-1].append(skew)
            
            fs = fs_list[i_exp]
            k_arr = np.arange(1, 2*fs)

            coefficients_2Dlist[-1].append(mre.coefficients(act,
                                           k_arr, dt=1/fs*1000, numboot=0,
                                           method='ts').coefficients)
            #raw_coefficients_2Dlist[-1].append(mre.coefficients(Fc,
            #                               k_arr, dt=1/fs*1000, numboot=0,
            #                               method='ts').coefficients) 
        #bright_2Dlist[-1] = np.array(bright_2Dlist[-1])
        #skew_2Dlist[-1] = np.array(skew_2Dlist[-1])
        coefficients_2Dlist[-1] = np.array(coefficients_2Dlist[-1])
        #raw_coefficients_2Dlist[-1] = np.array(raw_coefficients_2Dlist[-1])


    tau_select_2Dlist = []

    f, axes = plt.subplots(3, len(fs_list), figsize = (15,11))
    #to be automated
    titles = ['injected: 15 Hz, 73 min\n(2019-11-14, J061)',
              'injected: 15 Hz, 73 min\n(2019-11-13, J063)',
              'injected: 15 Hz, 73 min\n(2019-11-15, J064)',
              'injected: 15 Hz, 73 min\n(2019-11-18, J065)']
    
    for i_exp in range(len(titles)):
        fps_cutoff = 1000* 1/fs_list[i_exp]
        dcnv_cutoff = 1000*tau_dcnv

        raw_upper_cutoff = 3000
        raw_lower_cutoff = 750

        axes_exp = [ax[i_exp] for ax in axes]
        

        ### gate cells by tau
        tau_select = np.nonzero((tau_2Dlist[i_exp] > fps_cutoff) &
                                (tau_2Dlist[i_exp] < dcnv_cutoff))[0]
    
        #tau_select_2Dlist.append(tau_select)
        """
        tau_select_8 = np.nonzero((tau_2Dlist[i_exp] < fps_cutoff) & 
                            (tau_2Dlist[i_exp] < dcnv_cutoff) &
                            (raw_tau_2Dlist[i_exp] > raw_lower_cutoff) &
                            (raw_tau_2Dlist[i_exp] < raw_upper_cutoff)
                                )[0]
        
        axes_exp[a].legend(loc='lower right', 
                           labels=['q5: {}'.format(tau_select.shape[0])],
                           fontsize=9)
        """
        #print(tau_select)
        print('tau select shape')
        print(tau_select.shape)
        print('coefficients 2D list[iexp] shape')
        print(np.asarray(coefficients_2Dlist[i_exp]).shape)

        ### get raw autocorrelation
        k_arr = np.arange(1, 2*fs_list[i_exp])
        time_arr = k_arr/fs_list[i_exp]*1000

        ### get dcnv autocorrelation
        k_arr = np.arange(1, 2*fs_list[i_exp])
        time_arr = k_arr/fs_list[i_exp]*1000

        ### dcnv autocorrelation plot
        a = 0

        num_cells = np.asarray(coefficients_2Dlist[i_exp][tau_select]).shape[0]
        for i_cells in range(num_cells):
            axes_exp[a].plot(time_arr, 
                    coefficients_2Dlist[i_exp][tau_select][i_cells],
                    color='lightgrey', alpha=0.25)
        axes_exp[a].plot(time_arr, 
                np.mean(coefficients_2Dlist[i_exp][tau_select],
                    axis=0), color='black', linewidth=3)
        axes_exp[a].set_xlabel('time difference [ms]')
        if i_exp == 0:
            axes_exp[a].set_ylabel('autocorrelation')
        axes_exp[a].set_title(titles[a])

        ### tau_whisker(tau_spont) plot
        a = 1
        
        axes_exp[a].set_ylim((0,1000))
        axes_exp[a].set_xlim((0,1000))
        axes_exp[a].set_xlabel("spont timescale [ms]")
        if i_exp == 0:
            axes_exp[a].set_ylabel('whisker timescale [ms]')
        axes_exp[a].plot(tau_2Dlist[i_exp][tau_select], whisker_tau_2Dlist[i_exp][tau_select], 
                '.', alpha=0.3)
        axes_exp[a].plot(np.arange(0,1000), np.arange(0,1000), 
                '--', color='darkgrey')
        
        ### tau_whisker(tau_spont) plot
        a = 2
        
        axes_exp[a].set_ylim((0,1000))
        axes_exp[a].set_xlim((0,1000))
        axes_exp[a].set_xlabel("spont first half timescale [ms]")
        if i_exp == 0:
            axes_exp[a].set_ylabel('spont second half timescale [ms]')
        axes_exp[a].plot(tau_firstHalf_2Dlist[i_exp][tau_select], tau_secondHalf_2Dlist[i_exp][tau_select], 
                '.', alpha=0.3)
        axes_exp[a].plot(np.arange(0,1000), np.arange(0,1000), 
                '--', color='darkgrey')
    
    np.save('../processed_data/tau_select_2Dlist.npy', tau_select_2Dlist)
    np.save('../processed_data/tau_2Dlist.npy', tau_2Dlist)
    np.save('../processed_data/whisker_tau_2Dlist.npy', whisker_tau_2Dlist)
    np.save('../processed_data/tau_firstHalf_2Dlist.npy', tau_firstHalf_2Dlist)
    np.save('../processed_data/tau_secondHalf_2Dlist.npy', tau_secondHalf_2Dlist)

    plt.tight_layout()
    plt.savefig('../reports/analyze_transients/gated_whiskerdata_tauSelect'+
            '_fps'+str(fps_cutoff)+'_dcnv'+'inf'+'.pdf')
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

def analyze_specific_experiment_brightness():
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

        snr = calc_brightness_mloidolt(dcnv_m, Fc_m, fs)[0]
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

    axes[0].imshow(im, cmap='viridis')

    axes[1].imshow(ops1[0]['meanImg'], cmap='viridis')
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[1], run)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[1], run)
    axes[1].set_title('mean image')


    axes[2].imshow(ops1[0]['meanImgE'], cmap='viridis')
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[2], run)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[2], run)
    axes[2].set_title('mean image enhanced')

    axes[3].imshow(ops1[0]['max_proj'], cmap='viridis')
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[3], run, -3, -3)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[3], run, -3, -3)
    axes[3].set_title('maximal projection')
    plt.tight_layout()

    axes[4].imshow(ops1[0]['Vcorr'], cmap='viridis')
    for i in indices_small_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:orange', axes[4], run, -3, -3)
    for i in indices_large_tau[:n_cells]:
        plot_circle_around_ROI(i, 'tab:pink', axes[4], run, -3, -3)
    axes[4].set_title('Correlation map')
    plt.tight_layout()
    plt.savefig('../reports/analyze_transients/weird_and_normal_cell_in_2019-12-13_J064_brightness.pdf')


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
    # this definition of max_i conflicts with the Fc_max def below
    #max_i = np.argmax(dcvn_smoothed)
    # will use the following one instead:
    max_i = np.argmax(dcvn_smoothed[:-20])
    # what's going on with the +20? gonna comment it out
    #Fc_max = scipy.ndimage.gaussian_filter1d(Fc[i_cell], 5)[max_i + 20]
    Fc_max = scipy.ndimage.gaussian_filter1d(Fc[i_cell], 5)[max_i + 20]
    time_to_plot = 25
    fs = 30
    len_to_plot = time_to_plot * fs
    x = np.arange(len_to_plot) / fs - 5
    if max_i > len(dcvn_smoothed) - len_to_plot:
        return

    transient = Fc[i_cell][max_i - 5*fs:max_i + 20*fs] / Fc_max
    transient = scipy.ndimage.gaussian_filter1d(transient, 3)
    print(x.shape)
    print(transient.shape)
    try:
        ax.plot(x, transient, color=color, alpha=alpha)
    except:
        print(i_cell)

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

def open_vape_fluerescence_mloidolt():
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
    run_list = ['2019-11-14_J061',
        '2019-11-14_J063',
        '2019-11-15_J064',
        '2019-11-18_J065']

    # %%

    # local path to behaviour pickle files
    # this takes a while to load so maybe should do some further caching in the future
    pkl_path = os.path.expanduser('/data.nst/share/data/rob_calcium_mice')

    flus = []
    whisker_stim_flus = []
    fpss = []
    runs = []
    for mouse in run_list:
        run_path = os.path.join(pkl_path, mouse, '{}.pkl'.format(mouse))
        with open(run_path, 'rb') as f:
            session_obj = pickle.load(f)
            try:
                photo_s_flu = session_obj.photostim_s.raw[0]
                print('has phtostim_s')
                #print(photo_s_flu.shape)
                print(session_obj.photostim_s.frames)
            except:
                pass
            try:                
                photo_r_flu = session_obj.photostim_r.raw[0]
                print('has photostimi_r')
                #print(photo_r_flu.shape)
                print(session_obj.photostim_r.frames)
            except:
                pass
            try:
                whisker_flu = session_obj.whisker_stim.raw[0]
                print('has whisker')
                #print(whisker_flu.shape)
                print(session_obj.whisker_stim.frames)
            except:
                pass
            try:
                spont_flu = session_obj.spont.raw[0]
                print('has spont')
                #print(spont_flu.shape)
                print(session_obj.spont.frames)
            except:
                pass
            flus.append(session_obj.spont.raw[0][:,session_obj.spont.frames])
            whisker_stim_flus.append(session_obj.whisker_stim.raw[0][:,session_obj.whisker_stim.frames])
            fpss.append(session_obj.spont.fps)
    run = False
    flu = flus[0]
    return flus, runs, fpss, whisker_stim_flus
