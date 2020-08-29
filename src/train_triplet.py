import model
import generate_data as gd
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import TripletLossLayer
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


#----------------TRIPLET MODEL-----------------#

def triplets_model(input_shape, network, include_top=False, pooling=None):

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    # Get the embedded values
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    # Connect the inputs with the outputs
    triplet_net = Model(inputs=[anchor_input,positive_input,negative_input],outputs=[loss_layer])
    return triplet_net

def train_triplet_model(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes):
    build_representation = model.embedding_network(num_frames, 30, w, d, dimensions=embedding_size)
    tri_model = triplets_model(input_shape=(num_frames, 30, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    tri_model.compile(loss=[None],optimizer=optimizer,sample_weight_mode="temporal")
    build_representation.summary()
    tri_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, paths.model_images + '/Triplet_Model.png', show_shapes=True)

    if reason != 6 or reason != 7:
        trained_history = tri_model.fit(x=gd.triplet_generator(batch_size, X_train, Y_train), y=None, batch_size=batch_size, epochs=no_epochs, 
                                    verbose=2, callbacks=None, validation_data=None, shuffle=True, class_weight=None, 
                                    sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
        # early stopping needed to be complete
        trained_history = tri_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=2, callbacks=None, 
                                validation_split=None, validation_data=(X_val, labels_val), shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    build_representation.save(paths.model_save + model_name + str(reason) + '_Triplet_Representation' + str(embedding_size))
    tri_model.save(paths.model_save + model_name + str(reason) + '_Triplet_Model' + str(embedding_size))
    with open(paths.history_save + model_name + str(reason) + '_Triplet_History' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)

if __name__ == "__main__":
    my_reason = 1
    X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes = model.return_parameters(my_reason)
    train_triplet_model(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes)