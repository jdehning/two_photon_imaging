import sys
sys.path.append("../../mre")
import os

#import pyfits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from suite2p import dcnv
from matplotlib import colors
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from help_functions import deconvolve

import mrestimator as mre


def open_data_npy():
    filename = "../data/Spontaneous/Spontaneous_neuCorrected_F.npy"
    data_spont = np.load(filename)
    print(data_spont.shape)
    for i, arr in enumerate(data_spont):
        mat = arr.reshape((25, 717))
        mre.full_analysis(mat,
                          targetdir="./output_mre",
                          title="first_test_neuron{}".format(i),
                          dt=1/30,
                          dtunit="s",
                          fitfuncs=["eo"],
                          tmin=0,
                          tmax=50,
                          saveoverview=True,
                          showoverview=True)



def plot_data_position():
    data_dir = "../data/Spontaneous/"
    transients = np.load(os.path.join(data_dir, "Spontaneous_neuCorrected_F.npy"))
    print(transients.shape)
    pos_x = np.load(os.path.join(data_dir, "Spontaneous_x_pix.npy"))
    pos_y = np.load(os.path.join(data_dir, "Spontaneous_y_pix.npy"))
    tau_list = []
    mre._logstreamhandler.setLevel('ERROR')
    for i, arr in enumerate(tqdm(transients[:])):
        output_handler = mre.full_analysis(arr,
                          targetdir="./output_mre",
                          title="first_test_neuron{}".format(i),
                          dt=1 / 30,
                          dtunit="s",
                          fitfuncs=["eo"],
                          tmin=0,
                          tmax=50,
                          saveoverview=False,
                          showoverview=False)
        tau = output_handler.fits[0].tau
        tau_list.append(tau)
    def median_pos(mat):
        pos = []
        for arr in mat:
            pos.append(np.median(arr))
        return pos
    plt.clf()
    pos_x = median_pos(pos_x[:])
    pos_y = median_pos(pos_y[:])
    print(tau_list)
    cm = plt.cm.get_cmap("inferno")
    sc = plt.scatter(pos_x, pos_y, c=tau_list, vmin=0.6, vmax=1.8, s=35, cmap=cm)
    plt.colorbar(sc)
    plt.show()


def open_fits():
    filename = "../data/Spontaneous/2019-03-01_R024_Spontaneous_t-004_Cycle00001_Ch3.tif"
    hdulist = pyfits.open(filename)

def plot_spikes():
    directory = "../data/Spontaneous/suite2p/plane0/"
    iscell = np.load(directory+"iscell.npy")
    spks = np.load(directory+"spks.npy")
    F = np.load(directory+"F.npy")
    Fneu = np.load(directory+"Fneu.npy")
    t = np.linspace(0, F.shape[1]/30, F.shape[1])
    print(iscell[0])
    plt.plot(t, spks[0]*10)
    plt.plot(t, (F-0.7*Fneu)[0], alpha=0.8)
    plt.show()
    return
    mask_iscell = iscell[:,1] > 0.5
    spks_iscell = spks[mask_iscell, :]
    tau_list = []
    k_arr = np.arange(1,50)
    coefficients=np.zeros_like(k_arr, dtype=float)
    for i, arr in enumerate(tqdm(spks_iscell[:])):
        coeff_res  = mre.coefficients(arr, k_arr)
        coefficients += coeff_res.coefficients
    plt.plot(coefficients/len(spks_iscell))
    plt.show()
        #tau = output_handler.fits[0].tau
        #tau_list.append(tau)
    #def median_pos(mat):
    #    pos = []
    #    for arr in mat:
    #        pos.append(

def test_deconv():
    directory = "../data/Spontaneous/suite2p/plane0/"

    # load traces and subtract neuropil
    iscell = np.load(directory + "iscell.npy")
    F = np.load(directory + "F.npy")
    Fneu = np.load(directory + "Fneu.npy")
    Fc = F - 1 * Fneu

    N = 3

    mask_iscell = iscell[:, 1] > 0.5
    Fc = Fc[mask_iscell, :]
    Fc= Fc[:]

    #Fc_conv = np.empty((Fc.shape[0], Fc.shape[1]-N+1))
    #for i in range(len(Fc)):
    #    Fc_conv[i] = np.convolve(Fc[i], np.ones((N,))/N, mode='valid')
    Fc_conv = Fc
    spks = deconvolve(Fc_conv, fs=30, tau=1.3)

    t = np.linspace(0, Fc_conv.shape[1]/30, Fc_conv .shape[1])

    print(iscell[0])
    plt.plot(t, spks[0] * 10)
    plt.plot(t, Fc_conv[0], alpha=0.8)
    plt.show()


    k_arr = np.arange(1, 40)
    coefficients = np.zeros_like(k_arr, dtype=float)
    for i, arr in enumerate(tqdm(spks[:])):
        coeff_res = mre.coefficients(arr, k_arr)
        coefficients += coeff_res.coefficients
    f = plt.figure(figsize=(4,3))
    t  = np.arange(0, len(coefficients))/30
    plt.plot(t, coefficients / len(spks))
    #plt.gca().set_yscale('log')
    #plt.gca().set_xscale('log')
    plt.xlabel("time interval $\Delta T$ [s]")
    plt.ylabel("autocorrelation")
    plt.tight_layout()
    plt.savefig("../reports/estimating_timescales/figures/packer_example_lin_pil1.pdf")
    plt.show()



def test_with_spike_rec():
    directory = "../data/spikefinder_train/"
    dataset = "8"
    calcium = np.genfromtxt(directory + dataset + '.train.calcium.csv', delimiter=",", skip_header=1).transpose()
    spikes = np.genfromtxt(directory + dataset + '.train.spikes.csv', delimiter=",", skip_header=1).transpose()
    t = np.arange(calcium.shape[1]) / 100.0
    #plt.plot(t, calcium[2])
    #plt.plot(t, spikes[2])
    #plt.show()

    spks_deconv = np.empty_like(spikes)*np.nan
    for i, arr in enumerate(tqdm(calcium)):
        spks_tmp = deconvolve(arr[~np.isnan(arr)][None,:], 100, tau=1.5)[0]
        spks_deconv[i,:len(spks_tmp)] = spks_tmp


    k_arr = np.arange(1, 100)
    coefficients = np.zeros_like(k_arr, dtype=float)
    for i, arr in enumerate(tqdm(spikes[:])):
        coeff_res = mre.coefficients(arr[~np.isnan(arr)], k_arr)
        coefficients += coeff_res.coefficients

    coefficients_deconv = np.zeros_like(k_arr, dtype=float)
    for i, arr in enumerate(tqdm(spks_deconv[:])):
        coeff_res = mre.coefficients(arr[~np.isnan(arr)], k_arr)
        coefficients_deconv += coeff_res.coefficients

    print(coefficients)
    f = plt.figure(figsize=(4.3,3))
    t = np.arange(0, len(coefficients))/100
    plt.plot(t, coefficients / len(spikes), label="from spikes, 'ground truth'")
    plt.plot(t, coefficients_deconv / len(spks_deconv)/8.5, label='from deconvolved calcium imaging')
    plt.xlabel("time interval $\Delta T$ [s]")
    plt.ylabel("autocorrelation (normed)")

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    plt.legend()
    plt.tight_layout()
    plt.savefig("../reports/estimating_timescales/figures/comparison3_normed.pdf")




    plt.show()

def open_data():
    import napari
    from ScanImageTiffReader import ScanImageTiffReader
    vol = ScanImageTiffReader(path).data(beg = 0, end=50000)
    vol = np.mean(vol.reshape((500,100, vol.shape[1], vol.shape[2])), axis=1)
    napari.show_image(vol)

def plot_map_tau():
    directory = "../data/Spontaneous/suite2p/plane0/"

    # load traces and subtract neuropil
    iscell = np.load(directory + "iscell.npy")
    F = np.load(directory + "F.npy")
    Fneu = np.load(directory + "Fneu.npy")
    Fc = F - 1 * Fneu
    stat = np.load(directory + 'stat.npy', allow_pickle=True)
    x_pos = np.array([np.median(neur["xpix"]) for neur in stat])
    y_pos = np.array([np.median(neur["ypix"]) for neur in stat])

    mask_iscell = iscell[:, 1] > 0.5
    Fc = Fc[mask_iscell, :]
    Fc = Fc[:]
    x_pos = x_pos[mask_iscell]
    y_pos = y_pos[mask_iscell]
    x_pos = x_pos[:]
    y_pos = y_pos[:]

    Fc_conv = Fc
    spks = deconvolve(Fc_conv, fs=30, tau=1.5)

    t = np.linspace(0, Fc_conv.shape[1] / 30, Fc_conv.shape[1])


    k_arr = np.arange(1, 40)
    taus = []
    for i, arr in enumerate(tqdm(spks[:])):
        coeff_res = mre.coefficients(arr, k_arr, dt=1/30*1000)
        plt.plot(coeff_res.coefficients)
        plt.show()
        fit_result = mre.fit(coeff_res, steps=np.arange(1,40))
        taus.append(fit_result.tau)
    cm = plt.cm.get_cmap("inferno")
    sc = plt.scatter(x_pos, y_pos, c=taus, vmin=0, vmax=300, s=35, cmap=cm,
                     norm=colors.PowerNorm(0.7, 10,300))
    plt.xlabel("x position")
    plt.ylabel("y position")
    cbar = plt.colorbar(sc)
    cbar.set_label("Timescale [ms]")
    plt.tight_layout()
    plt.savefig("../reports/estimating_timescales/figures/map_taus.pdf")
    plt.show()

def correlation_different_resolutions():
    cells = {1: [8,2,14,0],
             2: [15,11,87,1],
             3: [19,240,196,4],
             4: [16,51,124,2],
             5: [ 24,255,305,3],
             6: [46,None,None, 9],
             7: [26,438, None, 17],
             8: [28,250,793,6]}
    labels = ['60 Hz, 0.54$\mu$m, 30 min', '60 Hz, 1.37$\mu$m, 15 min', '30 Hz, 1.37$\mu$m, 15 min', '60 Hz, 0.27$\mu$m, 30 min',
              '15 Hz, 1.37$\mu$m, 30 min (by subsampling)', '10 Hz, 1.37$\mu$m, 30 min (by subsampling)', '7.5 Hz, 1.37$\mu$m, 30 min (by subsampling)',
              '5 Hz, 1.37$\mu$m, 30 min (by subsampling)']
    tau=1.5
    fs_list=[60,60,30,60,15,10,7.5,5]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-001', '2019-11-08_RL065_t-002', '2019-11-08_RL065_t-003', '2019-11-08_RL065_t-005']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs in zip(Fc_list, cells[cell], fs_list):
            if cell_num is not None:
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,:], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)

    subsampling_list = [15,10,7.5, 5]
    for i, subs in enumerate(subsampling_list):
        for i_cell, cell in enumerate(cells.keys()):
            if cells[cell][1] is not None and cells[cell][2] is not None:
                cell_num = cells[cell][1]
                Fc = Fc_list[1][cell_num][None,::round(60//subs)]
                spks1 = deconvolve(Fc, subs, tau=tau)[0]
                cell_num = cells[cell][2]
                Fc = Fc_list[2][cell_num][None,::round(30//subs)]
                spks2 = deconvolve(Fc, subs, tau=tau)[0]
                spks_2dlist[i_cell].append(np.concatenate((spks1, spks2)))
            else:
                spks_2dlist[i_cell].append(None)


    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(4,2, figsize= (16,20))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:gray']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)

                plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/different_zoom_levels/autocorrelation_plots2.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.25,0.25, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/different_zoom_levels/timescales2.pdf')
    plt.show()


def correlation_different_resolutions2():
    cells = {1: [8,2,14,0],
             2: [15,11,87,1],
             3: [19,240,196,4],
             4: [16,51,124,2],
             5: [ 24,255,305,3],
             6: [46,None,None, 9],
             7: [26,438, None, 17],
             8: [28,250,793,6]}
    labels = ['60 Hz, 0.54$\mu$m, 30 min', '60 Hz, 1.37$\mu$m, 15 min', '30 Hz, 1.37$\mu$m, 15 min', '60 Hz, 0.27$\mu$m, 30 min',
              '30 Hz, 1.37$\mu$m, 15 min (by subsampling 60 Hz, 1.37$\mu$m)', '15 Hz, 1.37$\mu$m, 15 min (by subsampling 60 Hz, 1.37$\mu$m)',
              '15 Hz, 1.37$\mu$m, 15 min (by subsampling 30 Hz, 1.37$\mu$m)']
    tau=1.5
    fs_list=[60,60,30,60,30,15,15]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-001', '2019-11-08_RL065_t-002', '2019-11-08_RL065_t-003', '2019-11-08_RL065_t-005']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs in zip(Fc_list, cells[cell], fs_list):
            if cell_num is not None:
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,:], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)

    subsampling_list = [30, 15, 15]
    for i, subs in enumerate(subsampling_list):
        for i_cell, cell in enumerate(cells.keys()):
            if cells[cell][1] is not None and cells[cell][2] is not None:
                if i in [0,1]:
                    cell_num = cells[cell][1]
                    Fc = Fc_list[1][cell_num][None,::round(60//subs)]
                    spks1 = deconvolve(Fc, subs, tau=tau)[0]
                else:
                    spks1 = np.array([])
                if i in [2]:
                    cell_num = cells[cell][2]
                    Fc = Fc_list[2][cell_num][None,::round(30//subs)]
                    spks2 = deconvolve(Fc, subs, tau=tau)[0]
                else:
                    spks2 = np.array([])
                spks_2dlist[i_cell].append(np.concatenate((spks1, spks2)))
            else:
                spks_2dlist[i_cell].append(None)


    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(3,2, figsize= (16,16))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:gray']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)

                plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/different_zoom_levels/autocorrelation_plots2.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.25,0.25, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/different_zoom_levels/timescales2.pdf')
    plt.show()

def correlation_different_resolutions3():
    cells = {1: [8,2,14,0],
             2: [15,11,87,1],
             3: [19,240,196,4],
             4: [16,51,124,2],
             5: [ 24,255,305,3],
             6: [46,None,None, 9],
             7: [26,438, None, 17],
             8: [28,250,793,6]}
    labels = ['60 Hz, 0.54$\mu$m, 30 min', '60 Hz, 1.37$\mu$m, 15 min', '30 Hz, 1.37$\mu$m, 15 min', '60 Hz, 0.27$\mu$m, 30 min']
    tau=1.5
    fs_list=[60,60,30,60]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-001', '2019-11-08_RL065_t-002', '2019-11-08_RL065_t-003', '2019-11-08_RL065_t-005']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs in zip(Fc_list, cells[cell], fs_list):
            if cell_num is not None:
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,:], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)

    subsampling_list = []
    for i, subs in enumerate(subsampling_list):
        for i_cell, cell in enumerate(cells.keys()):
            if cells[cell][1] is not None and cells[cell][2] is not None:
                if i in [0,1]:
                    cell_num = cells[cell][1]
                    Fc = Fc_list[1][cell_num][None,::round(60//subs)]
                    spks1 = deconvolve(Fc, subs, tau=tau)[0]
                else:
                    spks1 = np.array([])
                if i in [2]:
                    cell_num = cells[cell][2]
                    Fc = Fc_list[2][cell_num][None,::round(30//subs)]
                    spks2 = deconvolve(Fc, subs, tau=tau)[0]
                else:
                    spks2 = np.array([])
                spks_2dlist[i_cell].append(np.concatenate((spks1, spks2)))
            else:
                spks_2dlist[i_cell].append(None)


    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(4,2, figsize= (16,20))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:gray']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)

                plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/different_zoom_levels/autocorrelation_plots_origin.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.2,0.2, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/different_zoom_levels/timescales_origin.pdf')
    plt.show()

def correlation_different_resolutions4():
    cells = {1: [8,2,14,0],
             2: [15,11,87,1],
             3: [19,240,196,4],
             4: [16,51,124,2],
             5: [ 24,255,305,3],
             6: [46,None,None, 9],
             7: [26,438, None, 17],
             8: [28,250,793,6]}
    labels = ['60 Hz, 0.54$\mu$m, 30 min', '30 Hz, 0.54$\mu$m, 30 min (downsampled)', '20 Hz, 0.54$\mu$m, 30 min (downsampled)',
              '15 Hz, 0.54$\mu$m, 30 min (downsampled)', '10 Hz, 0.54$\mu$m, 30 min (downsampled)', '7.5 Hz, 0.54$\mu$m, 30 min (downsampled)',
              '5 Hz 0.54$\mu$m, 30 min (downsampled)']
    tau=1.5
    fs_list=[60,30,20,15,10,7.5,5]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-001']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs in zip(Fc_list, cells[cell], fs_list):
            if cell_num is not None:
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,:], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)

    subsampling_list = [30,20,15,10,7.5,5]
    for i, subs in enumerate(subsampling_list):
        for i_cell, cell in enumerate(cells.keys()):
            if cells[cell][0] is not None:
                cell_num = cells[cell][0]
                Fc = Fc_list[0][cell_num][None,::round(60//subs)]
                spks1 = deconvolve(Fc, subs, tau=tau)[0]
                spks_2dlist[i_cell].append(spks1)
            else:
                spks_2dlist[i_cell].append(None)


    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(4,2, figsize= (16,20))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:gray']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    to_plot = [0,1,3,4]
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)
                if i in to_plot:
                    plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/different_zoom_levels/autocorrelation_plots_tempSubs1.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.25,0.25, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/different_zoom_levels/timescales_tempSubs1.pdf')
    plt.show()

def correlation_different_resolutions5():
    cells = {1: [8,2,14,0],
             2: [15,11,87,1],
             3: [19,240,196,4],
             4: [16,51,124,2],
             5: [ 24,255,305,3],
             6: [46,None,None, 9],
             7: [26,438, None, 17],
             8: [28,250,793,6]}

    labels = ['60 Hz, 1.37$\mu$m, 15 min', '30 Hz, 1.37$\mu$m, 15 min (downsampled)', '20 Hz, 1.37$\mu$m, 15 min (downsampled)',
              '15 Hz, 1.37$\mu$m, 15 min (downsampled)', '10 Hz, 1.37$\mu$m, 15 min (downsampled)', '7.5 Hz, 1.37$\mu$m, 15 min (downsampled)',
              '5 Hz 1.37$\mu$m, 15 min (downsampled)']

    tau=1.5
    fs_list=[60,30,20,15,10,7.5,5]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-002']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs in zip(Fc_list, cells[cell][1:], fs_list):
            if cell_num is not None:
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,:], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)

    subsampling_list = [30,20,15,10,7.5,5]
    for i, subs in enumerate(subsampling_list):
        for i_cell, cell in enumerate(cells.keys()):
            if cells[cell][1] is not None:
                cell_num = cells[cell][1]
                Fc = Fc_list[0][cell_num][None,::round(60//subs)]
                spks1 = deconvolve(Fc, subs, tau=tau)[0]
                spks_2dlist[i_cell].append(spks1)
            else:
                spks_2dlist[i_cell].append(None)


    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(4,2, figsize= (16,20))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:gray']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    to_plot = [0,1,3,4]
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)

                if i in to_plot:
                    plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/different_zoom_levels/autocorrelation_plots_tempSubs2.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.25,0.25, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/different_zoom_levels/timescales_tempSubs2.pdf')
    plt.show()

def correlation_spatial_subsampling():
    #cells = {1: [8,0,1,1,8,8],
    #         2: [15,4,3,3,15,15],
    #         3: [19,23,19,18,19,19],
    #         4: [16,40,88,75,16,16],
    #        5: [24,25,43,46,24,24],
    #         6: [46,58,48,109,46,46],
    #         7: [26,41,37,41,26,26],
    #         8: [28,45,31,37,28,28]}
    cells = {1: [8,0,1,1,1,2,2],
             2: [15,4,3,3,2,1,11],
             3: [19,23,19,18,18,19,240],
             4: [16,40,88,75,85,22,51],
             5: [24,25,43,46,38,39,255],
             6: [46,58,48,109,44,42,None],
             7: [26,41,37,41,39,40,438],
             8: [28,45,31,37,29,25,250]}

    labels = ['60 Hz, 0.54x0.54$\mu$m, 30 min', '60 Hz, 0.54x1.08$\mu$m, 30 min (subs.)', '60 Hz, 1.08x1.08$\mu$m, 30 min (subs.)',
              '60 Hz, 1.62x1.62$\mu$m, 30 min (subs.)', '60 Hz, 2.16x2.16$\mu$m, 30 min (subs.)', '60 Hz, 2.7x2.7$\mu$m, 30 min (subs.)', '60 Hz, 1.37$\mu$m, 15 min']
    tau=1.5
    fs_list=[60,60,60,60,60,60,60]
    k_arr_list = []
    Fc_list= []
    directory = '/data.nst/jdehning/packer_data/calcium_subsampled/'
    subpath_F = 'suite2p/plane0/F.npy'
    subpath_Fneu = 'suite2p/plane0/Fneu.npy'
    paths = ['2019-11-08_RL065_t-001_1x1','2019-11-08_RL065_t-001_1x2','2019-11-08_RL065_t-001_2x2','2019-11-08_RL065_t-001_3x3','2019-11-08_RL065_t-001_4x4','2019-11-08_RL065_t-001_5x5','2019-11-08_RL065_t-002']
    for path in paths:
        F = np.load(os.path.join(directory, path, subpath_F))
        Fneu = np.load(os.path.join(directory, path, subpath_Fneu))
        Fc_list.append(F - 0.7 * Fneu)
        del F, Fneu
    spks_2dlist = []
    for cell in cells.keys():
        spks_2dlist.append([])
        for Fc, cell_num,fs, i_rec in zip(Fc_list, cells[cell], fs_list, range(1000)):
            if cell_num is not None:
                if i_rec == 4:
                    beg, end = (0, Fc.shape[1] // 2)
                elif i_rec ==5:
                    beg, end = (Fc.shape[1] // 2, -1)
                else:
                    beg,end = (0, -1)
                spks_2dlist[-1].append(deconvolve(Fc[cell_num][None,beg:end], fs, tau=tau)[0])
            else:
                spks_2dlist[-1].append(None)




    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    f, axes_list = plt.subplots(4,2, figsize= (16,20))
    axes_list = [ax for axes in axes_list for ax in axes]
    colors  = ['tab:blue', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'k']

    tau_res_list2d = []

    def mult_coeff_res(coeff_res, mult):
        return coeff_res._replace(coefficients=coeff_res.coefficients* mult,
                                  stderrs= coeff_res.stderrs* mult if coeff_res.stderrs is not None else None)
    to_plot = [0,1,2,3,4,5]
    for cell in range(len(cells)):
        taus = []
        plot = mre.OutputHandler([], ax=axes_list[cell])
        tau_res_list2d.append([])
        for i in range(len(labels)):
            k_arr = np.arange(1, fs_list[i]*1)
            if spks_2dlist[cell][i] is not None:
                act = spks_2dlist[cell][i]
                print(act.shape)
                len_trial = round(40*fs_list[i])
                act = act[:-(len(act)%len_trial)].reshape((-1,len_trial))
                coeff_res = mre.coefficients(act, k_arr, dt=1 / fs_list[i] * 1000, numboot=500, method='ts')
                #norm = 1/coeff_res.coefficients[0]
                #coeff_res = mult_coeff_res(coeff_res, norm)
                #for j, bootstrapcrs in enumerate(coeff_res.bootstrapcrs):
                #    coeff_res.bootstrapcrs[j] = mult_coeff_res(bootstrapcrs, norm)
                #axes_list[cell].plot(k_arr/fs_list[i]*1000 ,coeff_res.coefficients, color=colors[i],
                #                     label = labels[i] if cell==0 else None)

                if i in to_plot:
                    plot.add_coefficients(coeff_res, color=colors[i], label = labels[i] if cell == 0 else None)
                tau_res_list2d[-1].append(mre.fit(coeff_res, fitfunc='exponentialoffset',numboot=500))
            else:
                tau_res_list2d[-1].append(None)
        #axes_list[cell].set_ylim(-0.3,1.2)
        axes_list[cell].set_title('cell {}'.format(cell))
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/spatial_subsampling/autocorrelation_plots_spatialSubs3.pdf')
    #axes_list[0].set_ylabel('Autocorrelation')
    plt.show()

    x_ticks = np.arange(len(cells))
    offsets = np.linspace(-0.25,0.25, len(labels))

    f, axes_list = plt.subplots(figsize=(10, 6))
    for i_ana, offset in enumerate(offsets):
        res_list = [tau_res_list2d[i_cell][i_ana] for i_cell in range(len(cells))]
        taus = np.array([res.tau if res is not None else None for res in res_list], dtype=np.float)
        yerr_lower = taus - np.array([res.tauquantiles[0] if res is not None else None for res in res_list], dtype=np.float)
        yerr_upper = np.array([res.tauquantiles[-1] if res is not None else None for res in res_list], dtype=np.float) - taus
        plt.errorbar(x_ticks+offset, y =taus, yerr = [yerr_lower, yerr_upper], color = colors[i_ana],
                     label = labels[i_ana],marker="x", elinewidth=1.5, capsize=2,
                          markersize=6, markeredgewidth=1,linestyle="")
    plt.ylim(0,400)
    plt.legend()
    plt.xlabel("Cell number")
    plt.ylabel('Timescale (ms)')
            #fit_result = mre.fit(coeff_res, steps=k_arr)
            #taus.append(fit_result.tau)
    plt.savefig('../reports/spatial_subsampling/timescales_spatialSubs3.pdf')
    plt.show()





if __name__ == "__main__":
    #open_data_npy()
    #plot_data_position()
    #plot_spikes()
    #test_deconv()
    #test_with_spike_rec()
    #plot_map_tau()
    #correlation_different_resolutions2()
    #correlation_different_resolutions3()
    #correlation_different_resolutions4()
    #correlation_different_resolutions5()
    #correlation_spatial_subsampling()
    pass
