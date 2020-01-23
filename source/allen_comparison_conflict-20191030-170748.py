import os, time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
hv.extension('bokeh')
import dask
import dask.distributed
from numba import jit
from scipy.ndimage import filters
import random
#cluster = dask.distributed.LocalCluster()
client = dask.distributed.Client('127.0.0.1:44489')


def preprocess(F,ops):
    sig = ops['sig_baseline']
    win = int(ops['win_baseline']*ops['fs'])
    if ops['baseline']=='maximin':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = filters.minimum_filter1d(Flow,    win)
        Flow = filters.maximum_filter1d(Flow,    win)
    elif ops['baseline']=='constant':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = np.amin(Flow)
    elif ops['baseline']=='constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = F - Flow

    return F

@jit(nopython=True)
def oasis_loop(ca, NT, v,w,t,l,s,g):
    for i in range(0,10):
        it = 0
        ip = 0

        while it<NT:
            v[ip], w[ip],t[ip],l[ip] = ca[it],1,it,1

            while ip>0:
                if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                    # violation of the constraint means merging pools
                    f1 = np.exp(g * l[ip-1])
                    f2 = np.exp(2 * g * l[ip-1])
                    wnew = w[ip-1] + w[ip] * f2
                    v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip]* f1) / wnew
                    w[ip-1] = wnew
                    l[ip-1] += l[ip]

                    ip += -1
                else:
                    break
            it += 1
            ip += 1

        s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])

        return s

def oasis_dcnv(F, ops):
    ca = F
    NT = F.shape[0]
    v = np.zeros((NT,))
    w = np.zeros((NT,))
    t = np.zeros((NT,), dtype='int')
    l = np.zeros((NT,))
    s = np.zeros((NT,))

    g = -1./(ops['tau'] * ops['fs'])


    return oasis_loop(ca, NT, v,w,t,l,s,g)

oasis_dcnv(np.array([1.0,2,3,0,1]), {'tau': 1.0, 'fs': 10.0})


@dask.delayed
def calc_corr(activity_arr):
    k_arr = np.arange(1,20)
    activity_arr = np.array(activity_arr, dtype="float32")
    mean = np.mean(activity_arr)
    corr_arr = np.empty_like(k_arr, dtype="float32")
    variance = np.mean((activity_arr - mean) ** 2)
    for i, step in enumerate(k_arr):
        x = activity_arr[0:-step]
        y = activity_arr[step:]
        nominator = np.mean((x - mean) * (y - mean))
        corr_coeff = nominator / variance
        corr_arr[i] = corr_coeff
    return corr_arr



def open_dir(path):
    ephys = []
    ophys = []
    dt = None
    for file_name in os.listdir(path):
        file = h5py.File(os.path.join(path, file_name), 'r')
        num_cells = file['iSpk'].shape[0]
        #print(file_name, num_cells)
        for i in range(num_cells):
            ephys.append(np.histogram(file['iSpk'][i,:], bins = file['iFrames'][i,:])[0])
            ophys.append(file['f_cell'][i,:])
        if dt is None:
            dt = file['dto'][0]
        else:
            if not dt == file['dto'][0]:
                print(dt, file['dto'][0])

    return ephys, ophys, dt

def oasis(F, ops):
    assert F.shape[0] == 1
    F = preprocess(F, ops)
    sp = oasis_dcnv(F[0,:], ops)
    # results = Parallel(n_jobs=num_cores)(delayed(oasis1t)(F[i, :], ops) for i in inputs)
    # collect results as numpy array
    return sp


@dask.delayed
def deconvolve(ophys_arr, tau, dt):
    fs = 1/dt
    baseline = 'maximin'  # take the running max of the running min after smoothing with gaussian
    sig_baseline = 10.0  # in bins, standard deviation of gaussian with which to smooth
    win_baseline = 60.0  # in seconds, window in which to compute max/min filters

    ops = {'tau': tau, 'fs': fs, 'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline}
    # get spikes
    spks = oasis(ophys_arr[None,:], ops)
    return spks

def make_spikes(ephys, dt):
    corr_list = []
    for act in ephys:
        corr_list.append(calc_corr(act))
    return corr_list

def make_calcium(ophys, dt, tau):
    corr_list = []
    for act in ophys:
        deconvolved = deconvolve(act, tau, dt)
        corr_list.append(calc_corr(deconvolved))
    return corr_list

def make_corr_arr(tau = 1.5):
    directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_lowzoom/"
    directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_highzoom"
    #directory = "/home/jdehning/tmp/Emx1-s_highzoom"
    start_time = time.time()
    ephys, ophys, dt = open_dir(directory)
    corr_o = client.compute(make_calcium(ophys, dt, tau))
    corr_e = client.compute(make_spikes(ephys, dt))
    corr_o = client.gather(corr_o)
    corr_e = client.gather(corr_e)
    print('elapsed time: {:.2f}s'.format(time.time()-start_time))
    return corr_e, corr_o

def make_corr_arr_single(ephys, ophys, dt, i, tau = 1.5):
    corr_o = client.compute(make_calcium(ophys[i][None,:], dt, tau))
    corr_e = client.compute(make_spikes(ephys[i][None,:], dt))
    corr_o = client.gather(corr_o)
    corr_e = client.gather(corr_e)
    #print('elapsed time: {:.2f}s'.format(time.time()-start_time))
    return corr_e[0], corr_o[0]


def test():
    tau = 1.5
    directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_lowzoom/"
    directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_highzoom"
    directory = "/home/jdehning/tmp/Emx1-s_highzoom"
    ephys, ophys, dt = open_dir(directory)
    ophys_deconvolved = [deconvolve(arr, tau, dt) for arr in ophys]

    coeff_res_e_list = []
    coeff_res_o_list = []
    for i in range(len(ephys)):
        coeff_res_e_d = calc_corr(ephys[i])
        coeff_res_e_list.append(coeff_res_e_d)
        coeff_res_o_d = calc_corr(ophys_deconvolved[i])
        coeff_res_o_list.append(coeff_res_o_d)
    res_dask = client.persist(coeff_res_e_list + coeff_res_o_list)
    coeff_res_e_list = [res.compute() for res in res_dask[:len(coeff_res_e_list)]]
    coeff_res_o_list = [res.compute() for res in res_dask[len(coeff_res_o_list):]]
    #for coeff_res_e, coeff_res_o in zip(coeff_res_e_list, coeff_res_o_list):
    #    plt.plot(coeff_res_e)
    #    plt.plot(coeff_res_o)
    #plt.show()

def hv_test():
    def clifford_equation(a, b, c, d, x0, y0):
        xn, yn = x0, y0
        coords = [(x0, y0)]
        for i in range(10000):
            x_n1 = np.sin(a * yn) + c * np.cos(a * xn)
            y_n1 = np.sin(b * xn) + d * np.cos(b * yn)
            xn, yn = x_n1, y_n1
            coords.append((xn, yn))
        return coords

    hv.opts.defaults(
        hv.opts.Curve(color='black'),
        hv.opts.Points(color='red', alpha=0.1, width=400, height=400))

    def clifford_attractor(a, b, c, d):
        return hv.Points(clifford_equation(a, b, c, d, x0=0, y0=0))

    clifford = hv.DynamicMap(clifford_attractor, kdims=['a', 'b', 'c', 'd'])
    clifford.redim.range(a=(-1.5, -1), b=(1.5, 2), c=(1, 1.2), d=(0.75, 0.8), x=(-2, 2), y=(-2, 2))
    clifford


if __name__ == "__main__":
    #test()
    #hv_test()
    make_corr_arr()