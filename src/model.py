import generate_data as gd
import math
import numpy as np
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2

# Ideally we automatically infer the labels and code up but in this case hard coded so all the time we get the same numeric value for the label

label_dict = {'ABEO_ARCA.npy': 1, 'ABEO_BATS.npy': 2, 'ABEO_EDGA.npy': 3, 'ABEO_EDGX.npy': 4, 'ABEO_NASDAQ.npy': 5, 'GOOG_ARCA.npy': 6, 'GOOG_BATS.npy': 7, 'GOOG_EDGA.npy': 8, 'GOOG_EDGX.npy': 9, 'GOOG_NASDAQ.npy': 10, 
'IBM_ARCA.npy': 11, 'IBM_BATS.npy': 12, 'IBM_EDGA.npy': 13, 'IBM_EDGX.npy': 14, 'IBM_NASDAQ.npy': 15, 'SPY_ARCA.npy': 16, 'SPY_BATS.npy': 17, 'SPY_EDGA.npy': 18, 'SPY_EDGX.npy': 19, 'SPY_NASDAQ.npy': 20, \
'VOD_ARCA.npy': 21, 'VOD_BATS.npy': 22, 'VOD_EDGA.npy': 23, 'VOD_EDGX.npy': 24, 'VOD_NASDAQ.npy': 25}

# Parameters - Modifiable for Various Configurations
reason = 0
embedding_size = 120
batch_size = 100
no_epochs = 10 # One iteration over all of the training data
nb_classes = 25

# Get Data and set classes 
X_train, Y_train, X_val, Y_val, model_name = gd.get_data(reason) 
Y_train_numeric = [label_dict[v] for v in Y_train]
Y_val_numeric = [label_dict[v] for v in Y_val]
labels_train = K.one_hot(Y_train_numeric, nb_classes)
labels_val = K.one_hot(Y_val_numeric, nb_classes)

# Parameters - Mostly static
num_frames = 500 # Number of LOB States our book contains
h = 30 # Book Depth 
w = 2 # Buy side and sell side
d = 2 # Price and Volume
steps_per_epoch_travelled = math.ceil(X_train.shape[0] / batch_size) # One gradient update, where in one step batch_size examples are processed

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

def embedding_network(frames, h, w, c, dimensions, include_top=False, pooling=None, classes=1):  
    inp = Input(shape=(frames, h, w, c))
    out = Lambda(lambda y: K.reshape(y, (-1, h, w, c)))(inp)
    num_features_cnn = np.prod(K.int_shape(out)[1:])
    out = Lambda(lambda y: K.reshape(y, (-1, frames, num_features_cnn)))(inp)
    out = BatchNormalization()(out)
    out = TCN(nb_filters=120, kernel_size=2, return_sequences=False, dilations=[1, 2, 4, 8, 16, 32, 64], 
              activation=tf.keras.activations.swish, nb_stacks=1, use_batch_norm=True, dropout_rate=0.08, kernel_initializer='he_uniform')(out)
    out = BatchNormalization()(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)    
    return Model(inputs=inp, outputs=out)

def return_parameters():
    return X_train, Y_train, X_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes