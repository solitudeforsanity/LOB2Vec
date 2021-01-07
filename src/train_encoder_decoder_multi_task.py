# UnboundLocalError: local variable 'logs' referenced before assignment - use model
import config
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
project_path = os.path.abspath(os.path.join('../..'))
sys.path.append(".")
import models.model as model
import evaluation
import data_preparation.data_generation as gd
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import CategoricalTruePositives, TripletLossLayer, QuadrupletLossLayer
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
AUTOTUNE = tf.data.experimental.AUTOTUNE

#----------------BASE MODEL-----------------#

def model_base(input_shape, nb_classes, network, include_top=False, pooling=None):
    input = Input(shape=input_shape, name='input')
    encoded = network(input)
    y1 = Dense(nb_classes, activation='softmax', bias_initializer=model.initialize_bias)(encoded)
    output = Model(inputs=[input],outputs=[y1])
    return output

def multi_task_model(input_shape, network, include_top=False, pooling=None):
    # Define model layers.
    input1 = Input(shape=(config.num_frames, 1), name='input_1')
    input2 = Input(shape=(config.num_frames, 1), name='input_2')
    input3 = Input(shape=input_shape, name='input_3')
    input4 = Input(shape=input_shape, name='input_4')
    
    encoded = network([input1, input2, input3, input4])
    
    output = Model(inputs=[input1, input2, input3, input4], outputs=[encoded])
    plot_model(output, paths.model_images + '/_encoder_decoder_e2e.png', show_shapes=True)
    return output

def train_model_base(reason):
    tcn_model = model.encoder_decoder_simple()
    #tcn_model = multi_task_model(input_shape=(config.num_frames, config.h, config.w, config.d), network=build_representation)
    optimizer = Adam(lr = 0.00001)

    losses = {
	#"side": "sparse_categorical_crossentropy"
	#"action": "sparse_categorical_crossentropy",
   # "price_level": 'sparse_categorical_crossentropy'
   # "liquidity": "sparse_categorical_crossentropy"
     "y_lob_prices" : "mean_squared_error"
    }

    acc = {
	#"side": "sparse_categorical_accuracy"
	#"action": "sparse_categorical_accuracy",
    #"price_level": "sparse_categorical_accuracy"
   # "liquidity": "sparse_categorical_accuracy"
    "y_lob_prices" : "mean_squared_error"
    }

    precision = {
    "side": tf.keras.metrics.Precision(),
    "action": tf.keras.metrics.Precision(),
    "price_level": tf.keras.metrics.Precision(),
    "liquidity": tf.keras.metrics.Precision()
    }

    recall = {
    "side": tf.keras.metrics.Recall(),
    "action": tf.keras.metrics.Recall(),
    "price_level": tf.keras.metrics.Recall(),
    "liquidity": tf.keras.metrics.Recall()
    }

    tcn_model.compile(loss=losses, optimizer=optimizer, metrics=acc)
   # build_representation.summary()
    tcn_model.summary()
   # tcn_full_summary(build_representation, expand_residual_blocks=True)
   # plot_model(build_representation, paths.model_images + '/_convlstm_multi_.png', show_shapes=True)
    #build_representation.save(paths.model_save + str(reason) + '_TCN_CC_Representation' +  str(config.embedding_size))
    return tcn_model

def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)

def model_fit(tcn_model, training_gen, validation_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, reason, stock):
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)
    csv_logger = CSVLogger(paths.logs_save + model_name + str(reason) + '_convlstm_multi_' + str(config.embedding_size), append=True, separator=';')
    logdir = paths.logs_save + model_name + str(reason) + '_convlstm_multi_' + str(config.embedding_size)
    tensorboard_callback = TensorBoard(log_dir=logdir)
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    trained_history = tcn_model.fit(x=training_gen, y=None, batch_size=config.batch_size, epochs=config.no_epochs, verbose=1, callbacks=[tensorboard_callback, lr_decay],
                                   validation_split=0.0, validation_data=validation_gen, shuffle=True, class_weight=None,
                                   sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled*0.9, validation_steps=val_steps_per_epoch_travelled*0.9,
                                   validation_batch_size=config.batch_size, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

    #tcn_model.save(paths.model_save + model_name + str(reason) + '_TCN_CC_Representation' + str(config.embedding_size))
    #with open(paths.history_save + model_name + str(reason) + '_CC_History_' + str(embedding_size), 'wb') as file_pi:
     #   pickle.dump(trained_history.history, file_pi)

    evaluation.multi_loss(trained_history, config.no_epochs, model_name + str(config.num_frames) + '_convlstm_multi_' + stock)
    evaluation.multi_acc(trained_history, config.no_epochs, model_name + str(config.num_frames) + '_convlstm_multi_' + stock)
    return tcn_model

if __name__ == "__main__":
    my_reason = 0
    gen_type = 1

    for stock in config.stock_list:
        tcn_model = train_model_base(my_reason)
        robust_scaler = RobustScaler()
        min_max_scaler = MinMaxScaler(feature_range=(-1,1))
        # For large data you can add double for loop here and pass in the paths until data_cleanser to get data
        training_gen, validation_gen, testing_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, robust_scaler\
                                                                    = model.return_parameters(stock, my_reason, gen_type, robust_scaler)
        tcn_model = model_fit(tcn_model, training_gen, validation_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, my_reason, stock[:-11])
        evaluation.generate_metrics(tcn_model, testing_gen, True, stock, model_name, robust_scaler)
