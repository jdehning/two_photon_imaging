import os, fnmatch, sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from suite2p import dcnv
from matplotlib import colors
import xarray as xr
import dask
import dask.array
from dask.distributed import Client,LocalCluster
from ScanImageTiffReader import ScanImageTiffReader
from skimage.external.tifffile import TiffFile
import time

import suite2p

#suite2p.run_s2p

def save_dir(directory):
    directory = "/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/2019-11-08_RL065_t-001/suite2p"

    # load traces and subtract neuropil
    for plane in next(os.walk(directory))[1]:
        path = os.path.join(directory, plane,'')
        iscell = np.load(path + "iscell.npy")
        F = np.load(path + "F.npy")
        Fneu = np.load(path + "Fneu.npy")
        Fc = F - 1 * Fneu
        stat = np.load(path + 'stat.npy', allow_pickle=True)
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
    #spks = deconvolve(Fc_conv, fs=30, tau=1.5)

@dask.delayed
def create_arr(i):
    a = np.arange(10, dtype='int32') + 10*i
    return a

def create_dask_array_test(num_chunks):
    shape_chunk = (10,)
    chunks = []
    for i in range(num_chunks):
        chunk = create_arr(i)
        chunk = dask.array.from_delayed(chunk, shape_chunk, dtype='int32')
        chunks.append(chunk)
    full_array = dask.array.concatenate(chunks)
    return full_array



@dask.delayed
def read_chunk(reader, num, size_chunk):
    beg = size_chunk*num
    end = beg + size_chunk
    #vol = reader.data(beg=beg, end=end)
    vol = reader.asarray(key=slice(beg, end))
    print(beg)
    return np.array(vol, dtype='uint16')

def create_dask_array(reader, num_chunks, size_chunk):
    chunks = []

    shape = (len(reader), reader.pages[0].shape[0], reader.pages[0].shape[1])
    for i in range(num_chunks):
        chunk = read_chunk(reader, i, size_chunk)
        length_curr_chunk = size_chunk if not i == num_chunks-1 else shape[0]%size_chunk
        shape_chunk = (length_curr_chunk, shape[1], shape[2])
        chunk=dask.array.from_delayed(chunk, shape_chunk, dtype='uint16')
        print(i)
        chunks.append(chunk)
    full_array = dask.array.concatenate(chunks)
    return full_array


def save_tiff_to_xr():
    #cluster = LocalCluster(processes=False)
    #client = Client(cluster)
    size_chunk = 500
    path = "/data.nst/share/data/packer_calcium_mice/2019-11-08_RL065/2019-11-08_RL065_t-001/"
    filename = "2019-11-08_RL065_t-001_Cycle00001_Ch3.tif"
    subsampling_list = [(4,4), (5,5)]
    for subs in subsampling_list:
        with dask.config.set(scheduler='single-threaded'):
            with TiffFile(path + filename, fastij = False) as reader:
                length = len(reader)
                num_chunks = (length-1)//size_chunk+1
                dask_array = create_dask_array(reader, num_chunks, size_chunk)
                dask_array = dask_array[:,::subs[0],::subs[1]]
                #print(dask_array.shape)
                #print(dask_array)
                #ds = xr.Dataset({'activity':(['time', 'x','y'], dask_array)})
                #ds.to_netcdf('/scratch.local/jdehning/calcium_subsampled/'+'2019-11-08_RL065_t-001_Cycle00001_Ch3_1x2.nc')
                #ds.to_zarr('/scratch.local/jdehning/calcium_subsampled/'+'2019-11-08_RL065_t-001_Cycle00001_Ch3_1x2.zr')
                dask_array.to_hdf5('/data.nst/jdehning/packer_data/calcium_subsampled/'+'2019-11-08_RL065_t-001_Cycle00001_Ch3_{}x{}.hdf5'.format(*subs),'/x')

def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def save_tiff_to_hdf():
    #cluster = LocalCluster(processes=False)
    #client = Client(cluster)
    size_chunk = 500
    path = "/data.nst/share/data/packer_calcium_mice/2019-11-07_J061_t-003"
    save_dir = os.path.join('/data.nst/jdehning/packer_data',os.path.basename(os.path.normpath(path)))
    save_name = os.path.basename(os.path.normpath(path))
    filename = find_file('*.tif', path)
    if len(filename) > 1:
        raise RuntimeError("More than one tif found")
    else:
        filename = filename[0]

    with dask.config.set(scheduler='single-threaded'):
        with TiffFile(os.path.join(path,filename), fastij = False) as reader:
            os.makedirs(save_dir, exist_ok=True)
            length = len(reader)
            num_chunks = (length-1)//size_chunk+1
            dask_array = create_dask_array(reader, num_chunks, size_chunk)
            dask_array = dask_array[:dask_array.shape[0]//2,::,::]
            dask_array.to_hdf5(os.path.join(save_dir, save_name),'/x')


if __name__ == '__main__':\
    #save_dir('')
    save_tiff_to_hdf()