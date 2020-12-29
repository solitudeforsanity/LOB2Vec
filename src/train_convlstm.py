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
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
AUTOTUNE = tf.data.experimental.AUTOTUNE

#----------------BASE MODEL-----------------#

def single_task_model(input_shape, nb_classes, network, include_top=False, pooling=None):
    input = Input(shape=input_shape, name='input')
    encoded = network(input)
    y1 = Dense(nb_classes, activation='softmax', bias_initializer=model.initialize_bias)(encoded)
    output = Model(inputs=[input],outputs=[y1])
    return output

def multi_task_model(input_shape, nb_classes, network, include_top=False, pooling=None):
    # Define model layers.
    input = Input(shape=input_shape, name='input')
    encoded = network(input)
    y1 = Dense(units=nb_classes, activation='softmax', name='price_output')(encoded)

    y2 = Dense(units=nb_classes, activation='softmax', name='ptratio_output')(encoded)
    # Define the model with the input layer 
    # and a list of output layers
    model = Model(inputs=input, outputs=[y1, y2])
    return model

def train_model_base(reason):
    build_representation = model.convlstm_network()
    cnnlstm_model = single_task_model(input_shape=(config.num_frames, config.h, config.w, config.d), network=build_representation, nb_classes=config.nb_st_classes)
    optimizer = Adam(lr = 0.00001)
    cnnlstm_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer,
                     metrics=[metrics.CategoricalAccuracy(name='acc'),
                     CategoricalTruePositives(config.nb_st_classes, config.batch_size)])
    build_representation.summary()
    cnnlstm_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, paths.model_images + '/ConvLSTM_CC_Model.png', show_shapes=True)
    return cnnlstm_model

def model_fit(cnnlstm_model, training_gen, validation_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, reason, stock):
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)
    csv_logger = CSVLogger(paths.logs_save + model_name + str(reason) + '_ConvLSTM_CC_Representation' + str(config.embedding_size), append=True, separator=';')
    logdir = paths.logs_save + model_name + str(reason) + '_ConvLSTM_CC_Representation' + str(config.embedding_size)
    tensorboard_callback = TensorBoard(log_dir=logdir)

    trained_history = cnnlstm_model.fit(x=training_gen, y=None, batch_size=config.batch_size, epochs=config.no_epochs, verbose=1, callbacks=None,
                                   validation_split=0.0, validation_data=validation_gen, shuffle=True, class_weight=None,
                                   sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled*0.9, validation_steps=val_steps_per_epoch_travelled*0.9,
                                   validation_batch_size=config.batch_size, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

    evaluation.loss(trained_history, config.no_epochs, model_name + str(config.num_frames) + '_ConvLSTM_CC_' + stock)
    evaluation.acc(trained_history, config.no_epochs, model_name + str(config.num_frames) + '_ConvLSTM_CC_' + stock)
    return cnnlstm_model

if __name__ == "__main__":
    my_reason = 1
    gen_type = 0
    cnnlstm_model = train_model_base(my_reason)

    for stock in config.stock_list:
        training_gen, validation_gen, testing_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, \
                                                                    = model.return_parameters(stock, my_reason, gen_type)
        cnnlstm_model = model_fit(cnnlstm_model, training_gen, validation_gen, model_name, steps_per_epoch_travelled, val_steps_per_epoch_travelled, my_reason, stock)

        loss, acc, cp = cnnlstm_model.evaluate(training_gen, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
        print('Restored model, accuracy: {:5.2f}%'.format(loss))
        pred = cnnlstm_model.predict(testing_gen, steps=testing_gen.__len__())

        # To convert from one-hot to class labels
        predicted_class = tf.argmax(pred, axis=1)
        y_labels = np.empty((0, config.nb_st_classes))
        
        for i in range(0, testing_gen.__len__()):
            y_labels = np.append(y_labels, testing_gen.__getitem__(i)[1], axis=0)

        true_class = tf.argmax(y_labels, axis=1)
    
        cnf_matrix = confusion_matrix(true_class, predicted_class, labels=list(range(0,22)))
        evaluation.plot_confusion_matrix(cnf_matrix, config.nb_st_classes, model_name + str(config.num_frames) + '_ConvLSTM_CC_' + stock)  # doctest: +SKIP


