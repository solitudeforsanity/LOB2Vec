import sys
sys.path.append(".")

import config
import os
import paths
import numpy as np
import tensorflow as tf

from os.path import basename
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from skimage.util.shape import view_as_windows
from scipy.ndimage.interpolation import shift
from tensorflow.keras import Input, Model, metrics, backend as K

min_max_scaler = MinMaxScaler(feature_range=(0,1))

# Use the below for multiple time series
# https://stackoverflow.com/questions/61155779/merge-or-append-multiple-keras-timeseriesgenerator-objects-into-one
 # define model
    # https://stackoverflow.com/questions/47671732/keras-input-a-3-channel-image-into-lstm
    # https://www.kaggle.com/kmader/what-is-convlstm

def retrieve_cleansed_data(lob, y_df, z_df, filename, isnormalised, overlapping, robust_scaler):
    """
    Returns : 
        x_lob_states : Full LOB with stacked States
        x_lob_prices : LOB with prices only
        x_lob_qty : LOB with qty only
        x_mid_price : mid prices for current state
        x_spread : spread for current state

        y_lob_prices : Shifted Y Values for lob_prices
        y_single_task : Labels for single task 
        y_multi_task : Labels for multi task (Includes categorical labels for side, action, liquidity, price_levels as well as values for next mid price, spread)
    """
    # As evidenced by above, we can technically select all in the second axis as there is only 1 element. However,
    # because we need a 2d input we make it 0. The 3rd axis is side so we need this

    # Create all the preprocessing required for X Values
    samples, timesteps, h, w, d = len(lob), config.num_frames, config.h, config.w, config.d
    b_qty = lob['quantity'][:,0,0,:h]
    s_qty = lob['quantity'][:,0,1,:h]
    x_lob_qty = np.stack((b_qty, s_qty), axis=2)

    b_price = lob['price'][:,0,0,:h]
    s_price = lob['price'][:,0,1,:h]
    x_lob_price = np.stack((b_price, s_price), axis=2)

    b_ts = lob['timestamp'][:,0,0,:h]
    b_ts = np.diff(b_ts, axis=0, prepend=b_ts[0].reshape(1, h))
    s_ts = lob['timestamp'][:,0,1,:h]
    s_ts = np.diff(s_ts, axis=0, prepend=s_ts[0].reshape(1, h))
    x_lob_timestamp = np.stack((b_ts, s_ts), axis=2)
    x_lob_timestamp = x_lob_timestamp / np.timedelta64(1, 'us')

    x_spread = lob['price'][:,0,1,0] - lob['price'][:,0,0,0]
    x_mid_price = np.where(((lob['price'][:,0,1,0] > 0) & (lob['price'][:,0,0,0] > 0)), ((lob['price'][:,0,1,0] + lob['price'][:,0,0,0])/2),\
                  np.where(((lob['price'][:,0,1,0] > 0) & (lob['price'][:,0,0,0] == 0)), lob['price'][:,0,1,0], \
                  np.where(((lob['price'][:,0,0,0] > 0) & (lob['price'][:,0,1,0] == 0)), lob['price'][:,0,0,0], 0)))
    
    if isnormalised:
        quantile_transformer = QuantileTransformer()
      #  lob_qty = lob_qty.reshape(-1,1)
      #  lob_qty = robust_scaler.fit_transform(lob_qty)
       # lob_qty = lob_qty.reshape(samples, h, w)

        #x_spread = x_spread.reshape(-1, 1)
       # x_spread = robust_scaler.fit_transform(x_spread)
        #x_spread = x_spread.reshape(samples,)

       # x_mid_price = x_mid_price.reshape(-1, 1)
       # x_mid_price = robust_scaler.fit_transform(x_mid_price)
       # x_mid_price = x_mid_price.reshape(samples,)

        #price_stream = price_stream.reshape(-1, 1)
        #price_stream = robust_scaler.fit_transform(price_stream)
        #price_stream = price_stream.reshape(samples,)

       # lob_price = lob_price.reshape(-1,1)
       # lob_price = min_max_scaler.fit_transform(lob_price)
       # lob_price = lob_price.reshape(lob_n, h, w)

    # If Time of entry is necessary add lob_ts and make last dimension 3
    x_lob_states = np.dstack((x_lob_price))
    x_lob_states = x_lob_states.reshape(samples, h, w, d)

    # Setting Up all Y Values for Training 
    y_mid = x_mid_price.reshape(-1, 1)
    y_spread = x_spread.reshape(-1, 1)
    y_df = np.append(y_df, y_mid, axis=1)
    y_df = np.append(y_df, y_spread, axis=1)
    
    # Make spread and Mid Usable in LSTM Networks
    x_spread = x_spread[...,np.newaxis]
    x_mid_price = x_mid_price[...,np.newaxis]
    x_lob_price = x_lob_price[...,np.newaxis]

    # We use the num_frames for step count so that the windows are non-overlapping. We can also use view_as_blocks but the issue with this is that it
    # requires precise block splits. i.e: If block does not have enough data it will not make block

    if ((len(x_lob_states) - timesteps) < 0):
        return [], []
    else:
        # We are shifting Y values by one, since what we want from a state is the prediction of the action from that state. Without this shift, Y value
        # gives the action that achieved this current state. With this shift the last state will have action = 0, which is did nothing. 
        # TODO : SINGLE ACTION SPACE MAYBE STILL IN ROUNDED DOWN DECIMALS
        z_df_shifted = shift(z_df, -1, cval=0)
        y_df = y_df.astype(float)
        y_df_shifted = shift(y_df, shift=[-1,0], cval=0)
        # Rouding all ints up because shift would make a number like 22 to be 21.9999999991 and then in the astype(int) section in data_generation this now becomes an integer 21 and not 22. Therefore 
        # Here we first set the float to 22.0 by using rint and then in the data_generator now use astype to convert it to int 22
        y_df_shifted[:,0] = np.rint(y_df_shifted[:,0])
        y_df_shifted[:,1] = np.rint(y_df_shifted[:,1])
        y_df_shifted[:,2] = np.rint(y_df_shifted[:,2])
        y_df_shifted[:,3] = np.rint(y_df_shifted[:,3])

        if overlapping:
            x_lob_states = view_as_windows(x_lob_states,(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            x_lob_price = view_as_windows(x_lob_price,(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            x_mid_price = view_as_windows(x_mid_price,(timesteps,1))[...,0].transpose(0,2,1)
            x_spread = view_as_windows(x_spread,(timesteps,1))[...,0].transpose(0,2,1)
            
            y_lob_price = shift(x_lob_price, shift=[-1,0,0,0,0], cval=0)
            eps2zero(y_lob_price)
            y_df_shifted = y_df_shifted[timesteps-1:len(y_df_shifted)]
            z_df_shifted = z_df_shifted[timesteps-1:len(z_df_shifted)]
        else:
            x_lob_states = view_as_windows(x_lob_states,(timesteps,1,1,1), step=(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            x_lob_price = view_as_windows(x_lob_price,(timesteps,1,1,1), step=(timesteps,1,1,1))[...,0,0,0].transpose(0,4,1,2,3)
            x_mid_price = view_as_windows(x_mid_price,(timesteps,1), step=(timesteps,1))[...,0].transpose(0,2,1)
            x_spread = view_as_windows(x_spread,(timesteps,1), step=(timesteps,1))[...,0].transpose(0,2,1)

            y_lob_price = shift(x_lob_price, shift=[-1,0,0,0,0], cval=0)
            eps2zero(y_lob_price)
            y_df_shifted = y_df_shifted[timesteps-1::timesteps]
            z_df_shifted = z_df_shifted[timesteps-1::timesteps]
            
        # change price_stream to mid to work cor
        return x_lob_states, x_lob_price, x_lob_qty, x_mid_price, x_spread, y_df_shifted, z_df_shifted, y_lob_price, robust_scaler

def eps2zero(x, dtype=np.float64):
    """ this sets values < precision to zero in-place """
    x[np.abs(x) < np.finfo(dtype).precision] = 0

def convert_data_to_labels(stock_name, data_source, robust_scaler):
    """
    Generates time series data for a given stock for all days inside the folder
    stock_name : Name of the stock we want to generate TS for
    data_souce : souce path of the data
    """
    X_lob_states, X_lob_price, X_lob_qty, X_mid_price, X_spread, Y_df_shifted, Z_df_shifted, Y_lob_price = None, None, None, None, None, None, None, None

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

                x_lob_states, x_lob_price, x_lob_qty, x_mid_price, x_spread, y_df_shifted, z_df_shifted, y_lob_price, robust_scaler = retrieve_cleansed_data(npy_x, npy_y, npy_z, file, True, False, robust_scaler)

                if len(x_lob_states) > 0:
                    if X_lob_states is not None:
                        X_lob_states = np.append(X_lob_states, x_lob_states, axis=0)
                    else:
                        X_lob_states = x_lob_states

                if len(x_lob_price) > 0:
                    if X_lob_price is not None:
                        X_lob_price = np.append(X_lob_price, x_lob_price, axis=0)
                    else:
                        X_lob_price = x_lob_price

                if len(x_lob_qty) > 0:
                    if X_lob_qty is not None:
                        X_lob_qty = np.append(X_lob_qty, x_lob_qty, axis=0)
                    else:
                        X_lob_qty = x_lob_qty

                if len(x_mid_price) > 0:
                    if X_mid_price is not None:
                        X_mid_price = np.append(X_mid_price, x_mid_price, axis=0)
                    else:
                        X_mid_price = x_mid_price

                if len(x_spread) > 0:
                    if X_spread is not None:
                        X_spread = np.append(X_spread, x_spread, axis=0)
                    else:
                        X_spread = x_spread

                if len(y_df_shifted) > 0:
                    if Y_df_shifted is not None:
                        Y_df_shifted = np.append(Y_df_shifted, y_df_shifted, axis=0)
                    else:
                        Y_df_shifted = y_df_shifted

                if len(z_df_shifted) > 0:
                    if Z_df_shifted is not None:
                        Z_df_shifted = np.append(Z_df_shifted, z_df_shifted, axis=0)
                    else:
                        Z_df_shifted = z_df_shifted

                if len(y_lob_price) > 0:
                    if Y_lob_price is not None:
                        Y_lob_price = np.append(Y_lob_price, y_lob_price, axis=0)
                    else:
                        Y_lob_price = y_lob_price

    return X_lob_states, X_lob_price, X_lob_qty, X_mid_price, X_spread, Y_df_shifted, Z_df_shifted, Y_lob_price, robust_scaler


def generate_data_for_each_day_per_stock(stock_name, data_source, robust_scaler):   
    Y_len = 0
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

                lob_states, lob_price, lob_qty, y_df_shifted, z_df_shifted, spread, mid, robust_scaler = retrieve_cleansed_data(npy_x, npy_y, npy_z, file, True, True, robust_scaler)
                return lob_states, lob_price, y_df_shifted, z_df_shifted, spread, mid, robust_scaler


def return_y_len(stock_name, data_source, robust_scaler, Y_value='Y'):
    """
    Generates time series data for a given stock for all days inside the folder
    stock_name : Name of the stock we want to generate TS for
    data_souce : souce path of the data
    """
    Y = None

    for subdir, dirs, files in os.walk(data_source):
        for file in files:
            data_path = os.path.join(subdir, file)
            my_path = Path(data_path)
            date_path = my_path.parent.parent
            XorY = basename(my_path.parent)
            if XorY == Y_value and file == stock_name:
                npy_y = np.load(data_path, allow_pickle=True)
                if len(npy_y) > 0:
                    if Y is not None:
                        Y = np.append(Y, npy_y, axis=0)
                    else:
                        Y = npy_y
    return len(Y)