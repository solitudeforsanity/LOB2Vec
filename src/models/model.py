import config
import data_preparation.data_generation as gd
import math
import numpy as np
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization, Dropout, ConvLSTM2D, Bidirectional, Conv3D, Cropping3D, ZeroPadding3D, TimeDistributed, MaxPooling3D, UpSampling3D, Conv2D, LSTM, Concatenate, Reshape
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

def simple_starts_tcn():
    # Works for Spread and Mid Seperately reasonably well
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp1 = Input(shape=(frames, 1)) # Mid Price 
    inp2 = Input(shape=(frames, 1)) # Spread
    inp3 = Input(shape=(frames, h, w, d)) # Entire LOB

    out1 = LSTM(50, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=False)(out1)

    out2 = Lambda(lambda y: K.reshape(y, (-1, h, w, d)))(inp3)
    num_features_cnn = np.prod(K.int_shape(out2)[1:])
    out2 = Lambda(lambda y: K.reshape(y, (-1, frames, num_features_cnn)))(inp3)
    out2 = BatchNormalization()(out2)
    out2 = TCN(nb_filters=128, kernel_size=2, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32, 64],
              activation=tf.keras.activations.swish, nb_stacks=2, use_batch_norm=True, dropout_rate=0.08, kernel_initializer='he_uniform')(out2)
    out2 = Flatten()(out2)
    out2 = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out2)

    return Model(inputs=[inp1, inp2, inp3], outputs=[out2])

# Does not work for side prediction
def simple_starts():
    # Works for Spread and Mid Seperately reasonably well
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp1 = Input(shape=(frames, 1)) # Mid Price 
    inp2 = Input(shape=(frames, 1)) # Spread
    inp3 = Input(shape=(frames, h, w, d)) # Entire LOB

    out1 = LSTM(50, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=False)(out1)

    out = ConvLSTM2D(filters=128, kernel_size=(2,2), input_shape=(frames, h, w, d), return_sequences=True, recurrent_activation='hard_sigmoid', activation=tf.keras.activations.swish, padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_1')(inp3)
    out = BatchNormalization()(out)
    out = Dropout(0.3, name='dropout_1')(out)
    out = ConvLSTM2D(filters=64, kernel_size=(2,2), input_shape=(frames, h, w, d), return_sequences=False, recurrent_activation='hard_sigmoid', activation='tanh', padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_2')(out)
    #out = BatchNormalization()(out)
    #out = TimeDistributed(Flatten(name="flatten"))(out)
    #out = TimeDistributed(Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform'))(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)

    return Model(inputs=[inp1, inp2, inp3], outputs=[out1, out])

def simple_starts_mid_to_play():
    # Works for Mid reasonably well
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp1 = Input(shape=(frames, 1)) # Mid Price 
    inp2 = Input(shape=(frames, 1)) # Spread
    inp3 = Input(shape=(frames, h, w, d)) # Entire LOB

    out1 = LSTM(50, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=True)(out1)
    out1 = LSTM(50, recurrent_dropout=0.2, return_sequences=False)(out1)
    return Model(inputs=[inp1, inp2, inp3], outputs=out1)

def tcn_network():
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp1 = Input(shape=(frames, h, w, d))
    out1 = Lambda(lambda y: K.reshape(y, (-1, h, w, d)))(inp1)
    num_features_cnn = np.prod(K.int_shape(out1)[1:])
    out1 = Lambda(lambda y: K.reshape(y, (-1, frames, num_features_cnn)))(inp1)
    out1 = BatchNormalization()(out1)
    out1 = TCN(nb_filters=128, kernel_size=2, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32, 64],
              activation=tf.keras.activations.swish, nb_stacks=2, use_batch_norm=True, dropout_rate=0.08, kernel_initializer='he_uniform')(out1)
    out1 = Flatten()(out1)
    out1 = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out1)

    inp2 = Input(shape=(frames, 1))
    inp2 = BatchNormalization()(inp2)
    out2 = LSTM(128, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp2)
    out2 = TimeDistributed(Dense(5, activation=None))(out2)
    out2 = Flatten()(out2)
    
    inp3 = Input(shape=(frames, 1))
    inp3 = BatchNormalization()(inp3)
    out3 = LSTM(128, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp3)
    out3 = TimeDistributed(Dense(5, activation=None))(out3)
    out3 = Flatten()(out3)

    out = Concatenate()([out1, out2, out3])
    print(out.shape)
   # out = Reshape((-1, 1))(out)
    #out = LSTM(128, recurrent_dropout=0.2, return_sequences=True)(out)
   # out = TimeDistributed(Dense(20, activation=None))(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)
  
    return Model(inputs=[inp1, inp2, inp3], outputs=out)

def convlstm_network():
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp1 = Input(shape=(frames, h, w, d))
    out1 = ConvLSTM2D(filters=64, kernel_size=(2,2), input_shape=(frames, h, w, d), return_sequences=True, return_state=False, recurrent_activation='hard_sigmoid', activation=tf.keras.activations.swish, padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_1')(inp1)
    out1 = Flatten()(out1)
    out1 = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out1)

    inp2 = Input(shape=(frames, 1))
    out2 = LSTM(128, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp2)
    out2 = TimeDistributed(Dense(20, activation=None))(out2)
    out2 = Flatten()(out2)
    
    inp3 = Input(shape=(frames, 1))
    out3 = LSTM(128, input_shape=(config.batch_size, frames, 1), recurrent_dropout=0.2, return_sequences=True)(inp3)
    out3 = TimeDistributed(Dense(20, activation=None))(out3)
    out3 = Flatten()(out3)

    out = Concatenate()([out1, out2, out3])
    print(out.shape)
   # out = Reshape((-1, 1))(out)
    #out = LSTM(128, recurrent_dropout=0.2, return_sequences=True)(out)
   # out = TimeDistributed(Dense(20, activation=None))(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)

    return Model(inputs=[inp1, inp2, inp3], outputs=out)

def encoder_decoder_network():
    # https://github.com/sadari1/TumorDetectionDeepLearning?utm_source=catalyzex.com
    ## ENCODER STAGE
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    input_layer = Input(shape=(frames, h, w, d))
    conv_encoder1 = Conv3D(filters=5, kernel_size=2, strides=1, input_shape=(frames, h, w, d), activation = 'relu')(input_layer)
    pool_1 = MaxPooling3D(pool_size=1, strides=1)(conv_encoder1)
    conv_encoder2 = Conv3D(filters=5, kernel_size=1, strides=1, input_shape=(frames, h, w, d), activation = 'relu')(pool_1)
    pool_2 = MaxPooling3D(pool_size=1, strides=1)(conv_encoder2)
    encoder = Dense(5, activation='relu')(pool_2)

    ## DECODER STAGE
    upsample_1 = UpSampling3D(size=4)(encoder)
    conv_decoder1 = Conv3D(filters=5, kernel_size=2, strides=1, input_shape=(frames, h, w, d), activation = 'relu')(upsample_1)
    upsample_2 = UpSampling3D(size=4)(conv_decoder1)
    conv_decoder2 = Conv3D(filters=5, kernel_size=2, strides=1, input_shape=(frames, h, w, d), activation = 'relu')(upsample_2)

    #Flattening layer to match y_test's shape
    flat = Flatten()(conv_decoder2)
    output_layer = Dense(41, activation='softmax')(flat)
    return Model(inputs=input_layer, outputs=output_layer)

def convlstm_network_original():
    frames, h, w, d, embedding_size = config.num_frames, config.h, config.w, config.d, config.embedding_size
    inp = Input(shape=(frames, h, w, d))
    out = ConvLSTM2D(filters=128, kernel_size=(2,2), input_shape=(frames, h, w, d), return_sequences=True, recurrent_activation='hard_sigmoid', activation=tf.keras.activations.swish, padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_1')(inp)
    out = BatchNormalization()(out)
    out = Dropout(0.3, name='dropout_1')(out)
    out = ConvLSTM2D(filters=64, kernel_size=(2,2), input_shape=(frames, h, w, d), return_sequences=True, recurrent_activation='hard_sigmoid', activation='tanh', padding='same', kernel_initializer='glorot_uniform', name='convlstm2d_2')(out)
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

def return_parameters(stock, reason, gen_type, robust_scaler):
    # Get Data and set classes
    training_gen, validation_gen, testing_gen, model_name, train_size, val_size, robust_scaler = gd.get_generator_data(stock, reason, gen_type, robust_scaler)
    steps_per_epoch_travelled = math.floor(train_size / config.batch_size) # One gradient update, where in one step batch_size examples are processed
    val_steps_per_epoch_travelled = math.floor(val_size / config.batch_size) # One gradient update, where in one step batch_size examples are processed
    return training_gen, validation_gen, testing_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, robust_scaler
