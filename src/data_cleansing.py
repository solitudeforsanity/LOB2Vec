import os
import sys
import numpy as np
import pandas as pd
import paths
import model
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from pylab import *
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from skimage.util.shape import view_as_windows

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
    print(data_source)
    for subdir, dirs, files in os.walk(data_source):
        print(subdir)
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


def save_data(data_source, data_dest, datatype):
    """

    """
    X, Y = convert_data_to_labels(data_source, num_frames)
    np.save(data_dest + str(num_frames) + datatype + 'X.npy', X)
    np.save(data_dest + str(num_frames) + datatype + 'Y.npy', Y)
    print('Written To ' + str(data_dest + str(num_frames)))

# To run this you need high memory machine
def save_individual_files(data_source, save_location, frames):
    if not os.path.exists(str(save_location) + str(frames) + '_X/'):
        os.makedirs(str(save_location) + str(frames) + '_X/')
    X, y = convert_data_to_labels(data_source, frames)
    {np.save(save_location + str(frames) + '_X/' + str(k) + '.npy', v) for k, v in enumerate(X)}
    Y_numeric = [model.label_dict[v] for v in y]
    np.save(save_location + str(frames) + '_Y.npy', Y_numeric)
    print('Written To ' + str(save_location) + str(frames))

# Test Data
#save_data(paths.source_test_2017, paths.dest_2017, '_Test2017_')

#save_individual_files(paths.source_dev, paths.generator_dev, num_frames)
save_individual_files(paths.source_train_2016, paths.generator_train_2016, num_frames)
save_individual_files(paths.source_val_2016, paths.generator_val_2016, num_frames)
save_individual_files(paths.source_test_2016, paths.generator_test_2016, num_frames)

#save_individual_files(paths.source_train_2017, paths.generator_train_2017, num_frames)
#save_individual_files(paths.source_val_2017, paths.generator_val_2017, num_frames)
#save_individual_files(paths.source_test_2017, paths.generator_test_2017, num_frames)

