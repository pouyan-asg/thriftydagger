import torch
import os
import sys
import numpy as np
import pickle as pkl
import h5py


# print(torch.__version__)
# print(torch.cuda.is_available())

path = '/home/pouyan/phd/imitation_learning/thriftydagger/data/pouyan/ep_1754070270_7702003'
data = np.load(os.path.join(path, 'state_1754070275_7123857.npz'), allow_pickle=True)
print(data.keys())
print(data['states'].shape)
# print(data['states'][0])
print(data['action_infos'].shape)
print(data['successful'].shape)
print(data['env'])
# print(data)
print('---------------------')

# path_pkl = '/home/pouyan/phd/imitation_learning/thriftydagger/robosuite-30.pkl'
# data_pkl = pkl.load(open(path_pkl, 'rb'))
# print(data_pkl['obs'].shape)
# print(data_pkl['act'].shape)
# # print(data_pkl)
# print('---------------------')


# path_h5 = '/home/pouyan/phd/imitation_learning/robomimic/datasets/lift/ph/low_dim_v15.hdf5'
# with h5py.File(path_h5, 'r') as f:
#     print(f.keys())
#     # print(f['data'].keys())
#     # print(f[]['obs'].shape)
#     # print(f['act'].shape)
#     # print(f['next_obs'].shape)
#     # print(f['reward'].shape)
#     # print(f['terminal'].shape)
#     # print(f['env_name'])
#     # print(f['task_name'])
#     # print(f['meta'])