import numpy as np
import suite2p


ops_path = 'suite2p_60Hz.npy'

ops = np.load(ops_path, allow_pickle=True)
print(ops)