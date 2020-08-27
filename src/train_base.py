import model
import generate_data as gd
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import CategoricalTruePositives, TripletLossLayer, QuadrupletLossLayer
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

#----------------BASE MODEL-----------------#

def model_base(input_shape, network, nb_classes, include_top=False, pooling=None, ):
    input = Input(shape=input_shape, name='input')
    encoded = network(input)
    prediction = Dense(nb_classes, activation='softmax', bias_initializer=model.initialize_bias)(encoded)
    output = Model(inputs=[input],outputs=[prediction])
    return output

def train_model_base(X_train, Y_train, X_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes):
    build_representation = model.embedding_network(num_frames, h, w, d, dimensions=embedding_size)
    cc_model = model_base(input_shape=(num_frames, h, w, d), network=build_representation, nb_classes=nb_classes)
    optimizer = Adam(lr = 0.00006)
    cc_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, 
                    metrics=[metrics.MeanSquaredError(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                    tf.keras.metrics.CategoricalAccuracy(name='acc'),
                    CategoricalTruePositives(nb_classes, batch_size)])
    build_representation.summary()
    cc_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, 'CC_Model.png', show_shapes=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    if reason != 6 and reason != 7:
        trained_history = cc_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=[callback], 
                                validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
        trained_history = cc_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=[callback], 
                                validation_split=None, validation_data=(X_val, labels_val), shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    build_representation.save(paths.model_save + model_name + str(reason) + '_CC_Representation' + str(embedding_size))
    cc_model.save(paths.model_save + model_name + str(reason) + '_CC_Model' + str(embedding_size))
    with open(paths.history_save + model_name + str(reason) + 'CC_History' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)


if __name__ == "__main__":
    X_train, Y_train, X_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes = model.return_parameters()
    train_model_base(X_train, Y_train, X_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes)

