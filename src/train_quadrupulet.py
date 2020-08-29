import model
import generate_data as gd
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import QuadrupletLossLayer
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

#----------------QUADRUPLET MODEL-----------------#

def quadruplet_model(input_shape, network, margin=0.1, margin2=0.01):
        '''
        Define the Keras Model for training 
            Input : 
                input_shape : shape of input images
                network : Neural network to train outputing embeddings
                metricnetwork : Neural network to train the learned metric
                margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha1)
                margin2 : minimal distance between Anchor-Positive and Negative-Negative2 for the lossfunction (alpha2)

        '''
         # Define the tensors for the four input images
        anchor_input = Input(shape=input_shape, name="anchor_input")
        positive_input = Input(shape=input_shape, name="positive_input")
        negative_input = Input(shape=input_shape, name="negative_input") 
        negative2_input = Input(shape=input_shape, name="negative2_input")

        # Generate the encodings (feature vectors) for the four images
        encoded_a = network(anchor_input)
        encoded_p = network(positive_input)
        encoded_n = network(negative_input)
        encoded_n2 = network(negative2_input)

        #QuadrupletLoss Layer
        loss_layer = QuadrupletLossLayer(alpha=margin,beta=margin2,name='4xLoss')([encoded_a,encoded_p,encoded_n,encoded_n2])

        # Connect the inputs with the outputs
        network_train = Model(inputs=[anchor_input,positive_input,negative_input,negative2_input],outputs=loss_layer)

        # return the model
        return network_train

def train_quadruplet_model(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes):
    build_representation = model.embedding_network(num_frames, 30, w, d, embedding_size, dimensions=embedding_size)
    quad_model = quadruplet_model(input_shape=(num_frames, 30, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    quad_model.compile(loss=[None],optimizer=optimizer,sample_weight_mode="temporal")
    build_representation.summary()
    quad_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, paths.model_images + '/Quadruplet_Model.png', show_shapes=True)

    if reason != 6 or reason != 7:
        trained_history = quad_model.fit(x=gd.quadruplet_generator(batch_size, X_train, Y_train), y=None, batch_size=batch_size, epochs=no_epochs, 
                                    verbose=2, callbacks=None, validation_data=None, shuffle=True, class_weight=None, 
                                    sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
         # TO : YOU HAVE TO COMPLETE THIS CORRECTLY also early stopping needs to be complete
        trained_history = quad_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=2, callbacks=None, 
                                validation_split=None, validation_data=(X_val, labels_val), shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    build_representation.save(paths.model_save + model_name + str(reason) + '_Quadruplet_Representation' + str(embedding_size))
    quad_model.save(paths.model_save + model_name + str(reason) + '_Quadruplet_Model' + str(embedding_size))
    with open(paths.history_save + model_name + str(reason) + '_Quadruplet_History' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)

if __name__ == "__main__":
    my_reason = 1
    X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes = model.return_parameters(my_reason)
    train_quadruplet_model(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes)