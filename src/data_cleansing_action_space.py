import os
import sys

# Setting Paths
module_path = os.path.abspath(os.path.join('..'))
project_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)
if project_path not in sys.path:
    sys.path.append(project_path)
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
import logging
from pylab import *
from os.path import basename
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from skimage.util.shape import view_as_windows
from scipy.ndimage.interpolation import shift

# Parameters
num_frames = 100

def retrieve_cleansed_data(lob, y_df, width, filename):
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
    
    b_ts = lob['timestamp'][:,0,0,:]
    b_ts = np.diff(b_ts, axis=0, prepend=b_ts[0].reshape(1, 30))
    s_ts = lob['timestamp'][:,0,1,:]
    s_ts = np.diff(s_ts, axis=0, prepend=s_ts[0].reshape(1, 30))
    lob_ts = np.stack((b_ts, s_ts), axis=2)
    lob_ts = lob_ts / np.timedelta64(1, 'us')

    lob_states = np.dstack((lob_qty, lob_price, lob_ts))
    lob_states = lob_states.reshape(lob_n, h, w, 3)

    # We use the num_frames for step count so that the windows are non-overlapping. We can also use view_as_blocks but the issue with this is that it 
    # requires precise block splits. i.e: If block does not have enough data it will not make block
 
    if ((len(lob_states) - num_frames) < 0):
        return [], []
    else:
        # We are shifting Y values by one, since what we want from a state is the prediction of the action from that state. Without this shift, Y value
        # gives the action that achieved this current state. With this shift the last state will have action = 0, which is did nothing
        y_df_shifted = shift(y_df, -1, cval=0)
        
        # Use this to get non-overlapping windows. Y value calculation for this not complete
        lob_states = view_as_windows(lob_states,(width,1,1,1), step=(num_frames,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
        y_df_shifted = y_df_shifted[num_frames-1::num_frames]
        
        # Use this for overlapping windows. Y value calculation also complete
        #lob_states = view_as_windows(lob_states,(width,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
        #y_df_shifted = y_df_shifted[num_frames-1:len(y_df_shifted)]
        logging.error(lob_states.shape)
        logging.error(len(y_df_shifted))
        print(lob_states.shape)
        print(len(y_df_shifted))
        return lob_states, y_df_shifted
    

def convert_data_to_labels(data_source, frames):
    """
    
    """
    X = None
    Y = None
        
    for subdir, dirs, files in os.walk(data_source):
        for file in files:
            data_path = os.path.join(subdir, file)
            my_path = Path(data_path)
            date_path = my_path.parent.parent
            x_path = date_path / 'X' / file
            XorY = basename(my_path.parent)
            if XorY == 'Y':
                npy_y = np.load(data_path, allow_pickle=True)
                npy_x = np.load(x_path, allow_pickle=True)
                logging.error(x_path)
                logging.error(data_path)
		       
                print(data_path)
                print(x_path)
                x, y = retrieve_cleansed_data(npy_x, npy_y, frames, file)
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
    X, Y = convert_data_to_labels(data_source, frames)
    {np.save(save_location + str(frames) + '_X/' + str(k) + '.npy', v) for k, v in enumerate(X)}
    np.save(save_location + str(frames) + '_Y.npy', Y)
    logging.error('Written To ' + str(save_location) + str(frames))
    print('Written To ' + str(save_location) + str(frames))
    
# Test Data
#save_data(paths.source_test_2017, paths.dest_2017, '_Test2017_')

#save_individual_files(paths.source_train_dev, paths.generator_train_dev, num_frames)
#save_individual_files(paths.source_val_dev, paths.generator_val_dev, num_frames)
#save_individual_files(paths.source_test_dev, paths.generator_test_dev, num_frames)

save_individual_files(paths.source_val_2016, paths.generator_val_2016, num_frames)
save_individual_files(paths.source_train_2016, paths.generator_train_2016, num_frames)
#save_individual_files(paths.source_test_2016, paths.generator_test_2016, num_frames)

#save_individual_files(paths.source_train_2017, paths.generator_train_2017, num_frames)
#save_individual_files(paths.source_val_2017, paths.generator_val_2017, num_frames)
#save_individual_files(paths.source_test_2017, paths.generator_test_2017, num_frames)
