import sys
import os
module_path = os.path.abspath(os.path.join('..'))
project_path = os.path.abspath(os.path.join('../..'))
sys.path.append(".")

import src.config as config
import os
import src.paths as path
import numpy as np
import tensorflow as tf

from os.path import basename
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from skimage.util.shape import view_as_windows
from scipy.ndimage.interpolation import shift
from tensorflow.keras import Input, Model, metrics, backend as K

robust_scaler = RobustScaler()
min_max_scaler = MinMaxScaler(feature_range=(0,1))

# Use the below for multiple time series
# https://stackoverflow.com/questions/61155779/merge-or-append-multiple-keras-timeseriesgenerator-objects-into-one
 # define model
    # https://stackoverflow.com/questions/47671732/keras-input-a-3-channel-image-into-lstm
    # https://www.kaggle.com/kmader/what-is-convlstm

def retrieve_cleansed_data(lob, y_df, z_df, filename, isnormalised, overlapping, robust_scaler):
    """
    """
    # As evidenced by above, we can technically select all in the second axis as there is only 1 element. However,
    # because we need a 2d input we make it 0. The 3rd axis is side so we need this

    samples, timesteps, h, w, d = len(lob), config.num_frames, config.h, config.w, config.d
    b_qty = lob['quantity'][:,0,0,:]
    s_qty = lob['quantity'][:,0,1,:]
    lob_qty = np.stack((b_qty, s_qty), axis=2)

    b_price = lob['price'][:,0,0,:]
    s_price = lob['price'][:,0,1,:]
    lob_price = np.stack((b_price, s_price), axis=2)

    b_ts = lob['timestamp'][:,0,0,:]
    b_ts = np.diff(b_ts, axis=0, prepend=b_ts[0].reshape(1, 30))
    s_ts = lob['timestamp'][:,0,1,:]
    s_ts = np.diff(s_ts, axis=0, prepend=s_ts[0].reshape(1, 30))
    lob_timestamp = np.stack((b_ts, s_ts), axis=2)
    lob_timestamp = lob_timestamp / np.timedelta64(1, 'us')

    spread = lob['price'][:,0,1,0] - lob['price'][:,0,0,0]
    mid = (lob['price'][:,0,1,0] + lob['price'][:,0,0,0])/2
    
    if isnormalised:
        quantile_transformer = QuantileTransformer()

        lob_qty = lob_qty.reshape(-1,1)
        lob_qty = robust_scaler.fit_transform(lob_qty)
        lob_qty = lob_qty.reshape(samples, h, w)

        mid = mid.reshape(-1, 1)
        mid = robust_scaler.fit_transform(mid)
        mid = mid.reshape(samples,)

        spread = spread.reshape(-1, 1)
        spread = robust_scaler.fit_transform(spread)
        spread = spread.reshape(samples,)


       # lob_price = lob_price.reshape(-1,1)
       # lob_price = min_max_scaler.fit_transform(lob_price)
       # lob_price = lob_price.reshape(lob_n, h, w)

    # If Time of entry is necessary add lob_ts and make last dimension 3
    lob_states = np.dstack((lob_price))
    lob_states = lob_states.reshape(samples, h, w, d)
    lob_price = lob_price[...,np.newaxis]

    y_spread = spread.reshape(-1, 1)
    y_mid = mid.reshape(-1, 1)
    y_df = np.append(y_df, y_spread, axis=1)
    y_df = np.append(y_df, y_mid, axis=1)
    print(y_df.shape)
    
    spread = spread[...,np.newaxis]
    mid = mid[...,np.newaxis]

    # We use the num_frames for step count so that the windows are non-overlapping. We can also use view_as_blocks but the issue with this is that it
    # requires precise block splits. i.e: If block does not have enough data it will not make block

    if ((len(lob_states) - timesteps) < 0):
        return [], []
    else:
        # We are shifting Y values by one, since what we want from a state is the prediction of the action from that state. Without this shift, Y value
        # gives the action that achieved this current state. With this shift the last state will have action = 0, which is did nothing
        z_df_shifted = shift(z_df, -1, cval=0)
        y_df = y_df.astype(float)
        y_df_shifted = shift(y_df, shift=[-1,0], cval=0)

        if overlapping:
            lob_states = view_as_windows(lob_states,(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            lob_price = view_as_windows(lob_price,(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            spread = view_as_windows(spread,(timesteps,1))[...,0].transpose(0,2,1)
            mid = view_as_windows(mid,(timesteps,1))[...,0].transpose(0,2,1)

            y_df_shifted = y_df_shifted[timesteps-1:len(y_df_shifted)]
            z_df_shifted = z_df_shifted[timesteps-1:len(z_df_shifted)]
        else:
            lob_states = view_as_windows(lob_states,(timesteps,1,1,1), step=(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            lob_price = view_as_windows(lob_price,(timesteps,1,1,1), step=(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            spread = view_as_windows(spread,(timesteps,1), step=(timesteps,1))[...,0].transpose(0,2,1)
            mid = view_as_windows(mid,(timesteps,1), step=(timesteps,1))[...,0].transpose(0,2,1)

            y_df_shifted = y_df_shifted[timesteps-1::timesteps]
            z_df_shifted = z_df_shifted[timesteps-1::timesteps]
        return lob_states, lob_price, lob_qty, y_df_shifted, z_df_shifted, spread, mid

# define dataset
def convert_data_to_labels(stock_name, data_source, robust_scaler):
    """
    Generates time series data for a given stock for all days inside the folder
    stock_name : Name of the stock we want to generate TS for
    data_souce : souce path of the data
    """
    X = None
    P = None
    Q = None
    Y = None
    Z = None
    spread = None
    mid = None

    for subdir, dirs, files in os.walk(data_source):
        for file in files:
            data_path = os.path.join(subdir, file)
            my_path = Path(data_path)
            date_path = my_path.parent.parent
            x_path = date_path / 'X' / file
            z_path = date_path / 'Z' / file
            XorY = basename(my_path.parent)
            if XorY == 'Y' and file == stock_name:
                npy_y = np.load(data_path, allow_pickle=True)
                npy_x = np.load(x_path, allow_pickle=True)
                npy_z = np.load(z_path, allow_pickle=True)

                x, p, q, y, z, sp, md = retrieve_cleansed_data(npy_x, npy_y, npy_z, file, True, False, robust_scaler)
                if len(x) > 0:
                    if X is not None:
                        X = np.append(X, x, axis=0)
                    else:
                        X = x

                if len(p) > 0:
                    if P is not None:
                        P = np.append(P, p, axis=0)
                    else:
                        P = p

                if len(q) > 0:
                    if Q is not None:
                        Q = np.append(Q, q, axis=0)
                    else:
                        Q = q

                if len(y) > 0:
                    if Y is not None:
                        Y = np.append(Y, y, axis=0)
                    else:
                        Y = y

                if len(z) > 0:
                    if Z is not None:
                        Z = np.append(Z, z, axis=0)
                    else:
                        Z = z

                if len(sp) > 0:
                    if spread is not None:
                        spread = np.append(spread, sp, axis=0)
                    else:
                        spread = sp

                if len(md) > 0:
                    if mid is not None:
                        mid = np.append(mid, md, axis=0)
                    else:
                        mid = md

    return X, P, Y, Z, spread, mid

convert_data_to_labels('USM_NASDAQ.npy', path.source_train_dev, robust_scaler)

# define dataset
def convert_data_to_labels_days(stock_name, data_source):
    def gen(file_list, data_path, index, file, batch_size = config.batch_size):
        """
        Generates time series data for a given stock for all days inside the folder
        stock_name : Name of the stock we want to generate TS for
        data_souce : souce path of the data
        """
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))
        y_batch = np.zeros((self.batch_size, len(self.Y_values[0])))
        X = None
        P = None
        Q = None
        Y = None
        Z = None

        my_path = Path(data_path)
        date_path = my_path.parent.parent
        x_path = date_path / 'X' / file
        z_path = date_path / 'Z' / file
        XorY = basename(my_path.parent)
        if XorY == 'Y':
            npy_y = np.load(data_path, allow_pickle=True)
            npy_x = np.load(x_path, allow_pickle=True)
            npy_z = np.load(z_path, allow_pickle=True)

            x, p, q, y, z = retrieve_cleansed_data(npy_x, npy_y, npy_z, file, True, True)
        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = x[idx]
            y_batch[idx_arr] = y[idx]
        yield x, y, z
    return gen
