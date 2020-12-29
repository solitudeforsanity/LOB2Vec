import data_preparation.data_generation as gd
import math
import numpy as np
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2

def initialize_weights(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

# Single Task Learning 

# 1. TCN (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 2. LSTM (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 3. Encoder-Decoder (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 4. Passing in entire time series without any work done

# Multi Task Learning 

# 1. TCN (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 2. LSTM (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 3. Encoder-Decoder (10, 100, 500), normalised and unnormalised, all frames vs individual time series
# 4. Passing in entire time series without any work done