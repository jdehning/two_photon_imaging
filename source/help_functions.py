import numpy as np
import os

import mrestimator as mre
from suite2p.extraction import dcnv



def deconvolve(mat, fs, tau = 1.5):
    #neucoeff = 0.7  # neuropil coefficient
    # for computing and subtracting baseline
    baseline = 'maximin'  # take the running max of the running min after smoothing with gaussian
    sig_baseline = 10.0  # in bins, standard deviation of gaussian with which to smooth
    win_baseline = 60.0  # in seconds, window in which to compute max/min filters

    ops = {'tau': tau, 'fs': fs, #'neucoeff': neucoeff,
           'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline}
    # get spikes
    spks = dcnv.oasis(mat, ops)
    return spks


def get_Fc(folder):
    F = np.load(os.path.join(folder, 'F.npy'))
    Fneu = np.load(os.path.join(folder, 'Fneu.npy'))
    Fc = F - 0.7 * Fneu
    return Fc

def get_cell_nums(folder):
    iscell = np.load(os.path.join(folder, "iscell.npy"))
    return np.nonzero(iscell[:,0])

def deconvolve_Fc(Fc, fs, cell_num = None, tau=1.5):
    """
    Returns deconvolved calcium signal. Returns 1 dimensional array if cell_num is an integes, otherwise a two dimensional array
    """
    if cell_num is None:
        Fc = Fc
        output_ind = Ellipsis
    elif isinstance(cell_num, int):
        Fc = Fc[cell_num]
        output_ind = 0
    else:
        Fc = Fc[np.newaxis, cell_num]
        output_ind = Ellipsis
    deconvolved = deconvolve(Fc, fs, tau=tau)[output_ind]
    return deconvolved

def fit_tau(act, fs, numboot = 0, k_arr = None):
    if k_arr is None:
        k_arr = np.arange(1, fs * 1)
    coeff_res = mre.coefficients(act, k_arr, dt=1/fs* 1000, numboot=numboot, method='ts')
    tau_res = mre.fit(coeff_res, fitfunc='exponentialoffset', numboot=numboot)
    return tau_res.tau

def rolling_sum(x, n_bins, same_length = True):
    shape_ret = list(x.shape)
    shape_ret[-1] -= n_bins-1
    ret = np.zeros(shape_ret)
    for i in range(n_bins):
        ret += x[...,i:x.shape[-1] + 1 - n_bins + i]
    if same_length:
        shape_to_add = shape_ret
        shape_to_add[-1] = n_bins-1
        return np.concatenate([np.zeros(shape_to_add), ret], axis=-1)
    else:
        return ret


def calc_signal(act, n_bins, nth_largest):
    """
    Returns snr of the last axis
    :param act:
    :param n_bins:
    :param nth_largest:
    :return:
    """
    assert len(act.shape) in [1,2]
    act_roll_sum = rolling_sum(act,n_bins)
    first_index = Ellipsis if len(act.shape) == 1 else np.arange(act.shape[0])
    for i in range(nth_largest):
        i_max = np.argmax(act_roll_sum, axis=-1)
        max_val = np.copy(act_roll_sum[first_index,i_max])
        for i_bin in range(-n_bins+1, n_bins):
            remove_borders_ind = np.min([np.max([np.zeros_like(i_max),i_max+i_bin], axis=0), np.ones_like(i_max)*act.shape[-1]-1], axis=0)
            act_roll_sum[first_index, remove_borders_ind] -= max_val

    return max_val

def calc_snr_mloidolt(dcnv, Fc, framerate, thresh=0.02):
    print(framerate)
    print(dcnv.shape)
    print(Fc.shape)
    snr = np.zeros(dcnv.shape[0])
    for i_cells in range(dcnv.shape[0]):
        spike_starts = np.nonzero(dcnv[i_cells,:]/np.mean(dcnv[i_cells,:] > thresh))[0]
        spike_ends = spike_starts[:-framerate] + framerate
        spike_idx = np.hstack([np.arange(s,e) for s,e in zip(spike_starts, spike_ends)])
        silent_idx = np.setdiff1d(np.arange(dcnv.shape[1]), spike_idx)
        snr[i_cells] = np.max(Fc[i_cells,:]/np.std(Fc[i_cells,silent_idx]))
    print(snr.shape)
    return snr

