import os

import numpy as np
from suite2p.run_s2p import run_s2p

db = {
      'h5py': None, # a single h5 file
      'h5py_key': ['x'], # list of keys to use (they will be extracted in the order you give them)
      'look_one_level_down': False, # for h5 files, whether to use all files in same folder
      'data_path': [] # keep this empty!
    }

ops = {
        'batch_size': 100, # reduce if running out of RAM
        'fast_disk': '/scratch.local/jdehning/suite2p_tmp', # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
        #'save_path0': '/media/jamesrowland/DATA/plab/suite_2p', # stores results, defaults to first item in data_path
        'delete_bin': True, # whether to delete binary file after processing
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter': 15, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1.5, # this is the main parameter for deconvolution
        'fs': 60.,  # sampling rate (total across planes)
        # output settings
        'save_mat': False, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True, # whether to register data
        'nimg_init': 200, # subsampled frames for finding reference image
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        # cell detection settings
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5001, # max number of binned frames for the SVD
        'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
      }

def run_subsampling():
    for dataset in ['4x4', '5x5']:
        basepath = '/data.nst/jdehning/packer_data/calcium_subsampled/'
        dic_prop = {'2x1': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_2x1.hdf5', '2019-11-08_RL065_t-001_2x1', 16],
                    '2x2': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_2x2.hdf5', '2019-11-08_RL065_t-001_2x2', 13],
                    '3x3': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_3x3.hdf5', '2019-11-08_RL065_t-001_3x3', 9],
                    '1x2': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_1x2.hdf5', '2019-11-08_RL065_t-001_1x2', 9],
                    '4x4': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_4x4.hdf5', '2019-11-08_RL065_t-001_4x4', 6],
                    '5x5': ['2019-11-08_RL065_t-001_Cycle00001_Ch3_5x5.hdf5', '2019-11-08_RL065_t-001_5x5', 5]}
        ops['diameter'] = dic_prop[dataset][2]
        db['h5py'] = basepath + dic_prop[dataset][0]
        db['save_path0'] = basepath + dic_prop[dataset][1]
        run_s2p(ops=ops, db=db)

def run_suite2p(path_hdf5, diameter):
    save_path = os.path.dirname(path_hdf5)
    ops['diameter'] = diameter
    db['h5py'] = path_hdf5
    db['save_path0'] = save_path
    run_s2p(ops=ops, db=db)


if __name__ == '__main__':
    #run_subsampling()
    run_suite2p('/data.nst/jdehning/packer_data/2019-11-07_J061_t-003/2019-11-07_J061_t-003.hdf5', 11)
