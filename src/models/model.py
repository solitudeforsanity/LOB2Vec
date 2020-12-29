import config
import data_preparation.data_generation as gd
import math
import numpy as np
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization, Dropout, ConvLSTM2D, Bidirectional, Conv3D, Cropping3D, ZeroPadding3D, TimeDistributed
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

def tcn_network(include_top=False, pooling=None, classes=1):
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp = Input(shape=(frames, h, w, d))
    out = Lambda(lambda y: K.reshape(y, (-1, h, w, d)))(inp)
    num_features_cnn = np.prod(K.int_shape(out)[1:])
    out = Lambda(lambda y: K.reshape(y, (-1, frames, num_features_cnn)))(inp)
    out = BatchNormalization()(out)
    out = TCN(nb_filters=128, kernel_size=2, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32, 64],
              activation=tf.keras.activations.swish, nb_stacks=2, use_batch_norm=True, dropout_rate=0.08, kernel_initializer='he_uniform')(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)
    return Model(inputs=inp, outputs=out)

def convlstm_network():
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp = Input(shape=(frames, h, w, d))
    out =  ConvLSTM2D(filters=64, kernel_size=(3,3), input_shape=(frames, h, w, d), return_sequences=True, recurrent_activation='hard_sigmoid', activation=tf.keras.activations.swish, padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_1')(inp)
    out = BatchNormalization()(out)
    out = Dropout(0.3, name='dropout_1')(out)
    out = ConvLSTM2D(filters=64, kernel_size=(3,3), input_shape=(frames, h, w, d), return_sequences=True, recurrent_activation='hard_sigmoid', activation='tanh', padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_2')(out)
    out = BatchNormalization()(out)
    #out = TimeDistributed(Flatten(name="flatten"))(out)
    #out = TimeDistributed(Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform'))(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)
    return Model(inputs=inp, outputs=out)

def transformer_network():
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp = Input(shape=(frames, h, w, d))

    '''Initialize time and transformer layers'''
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    x = Input(shape=(frames, h, w, d))
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(x)

    return Model(inputs=x, outputs=out)

def return_parameters(stock, reason, gen_type):
    # Get Data and set classes
    training_gen, validation_gen, testing_gen, model_name, train_size, val_size = gd.get_generator_data(stock, reason, gen_type)
    steps_per_epoch_travelled = math.floor(train_size / config.batch_size) # One gradient update, where in one step batch_size examples are processed
    val_steps_per_epoch_travelled = math.floor(val_size / config.batch_size) # One gradient update, where in one step batch_size examples are processed
    return training_gen, validation_gen, testing_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled
