import os
import sys

# Setting Paths
module_path = os.path.abspath(os.path.join('..'))
project_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import pandas as pd
import paths
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from pylab import *
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from skimage.util.shape import view_as_windows

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

# Parameters
num_frames = 500

def retrieve_cleansed_data(lob, width, filename):
    min_max_scaler = MinMaxScaler(feature_range=(0,50))
    quantile_transformer = QuantileTransformer()
    
    # As evidenced by above, we can technically select all in the second axis as there is only 1 element. However, 
    # because we need a 2d input we make it 0. The 3rd axis is side so we need this
    lob_qty_buy = pd.DataFrame(lob['quantity'][:,0,0,0:20])
    lob_qty_buy = lob_qty_buy.replace(0, np.NaN)
    
    lob_qty_sell = pd.DataFrame(lob['quantity'][:,0,1,0:20])
    lob_qty_sell = lob_qty_sell.replace(0, np.NaN)
  
    lob_n, d, w, h = lob['quantity'].shape
    b_qty = lob['quantity'][:,0,0,:]
    s_qty = lob['quantity'][:,0,1,:]
    lob_qty = np.stack((b_qty, s_qty), axis=2)

    lob_qty = lob_qty.reshape(-1,1)
    lob_qty = min_max_scaler.fit_transform(lob_qty)
    lob_qty = lob_qty.reshape(lob_n, h, w)
    
    b_price = lob['price'][:,0,0,:]
    s_price = lob['price'][:,0,1,:]
    lob_price = np.stack((b_price, s_price), axis=2)

    lob_price = lob_price.reshape(-1,1)
    lob_price = min_max_scaler.fit_transform(lob_price)
    lob_price = lob_price.reshape(lob_n, h, w)

    lob_states = np.dstack((lob_qty, lob_price))
    lob_states = lob_states.reshape(lob_n, h, w, 2)

    # We use the num_frames for step count so that the windows are non-overlapping. We can also use view_as_blocks but the issue with this is that it 
    # requires precise block splits. i.e: If block does not have enough data it will not make block
    print(lob_states.shape)
    if ((len(lob_states) - num_frames) < 0):
        return [], []
    else:
        lob_states = view_as_windows(lob_states,(width,1,1,1), step=(num_frames,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
        labels = np.full(len(lob_states), filename)
        return lob_states, labels


def convert_data_to_labels(data_source, frames):
    X = None
    Y = None
    for subdir, dirs, files in os.walk(data_source):
        for file in files:
            data_path = os.path.join(subdir, file)
            print(data_path)
            npy = np.load(data_path)
            x, y = retrieve_cleansed_data(npy, frames, file)
            if len(x) > 0:
                if X is not None:
                    X = np.append(X, x, axis=0)
                else:
                    X = x
                    
            if len(y) > 0:    
                if Y is not None:
                    Y = np.append(Y, y, axis=0)
                else:
                    Y = y
    return X, Y


def save_data(data_source, data_dest):
    X, Y = convert_data_to_labels(data_source, num_frames)

    np.save(data_dest + '/' + str(num_frames) + '_X.npy', X)
    np.save(data_dest + '/' + str(num_frames) + '_Y.npy', Y)

    print('Written To ' + str(data_dest + '/' + str(num_frames)))

# Test

# 2016
#save_data(paths.source_2016, paths.dest_2016)

# 2017
save_data(paths.source_2017, paths.dest_2017)

