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

reason = 3 

# Get Data and set classes 
X_train, Y_train, X_test, Y_test, model_name = gd.get_data(reason) 
unique_Y_train_classes = list(set(Y_train))
Y_train_dict = dict(zip(unique_Y_train_classes, list(range(1,len(unique_Y_train_classes)+1))))
Y_train_numeric = [Y_train_dict[v] for v in Y_train]

# Parameters - Mostly static
num_frames = 500 # Number of LOB States our book contains
h = 30 # Book Depth 
w = 2 # Buy side and sell side
d = 2 # Price and Volume
nb_classes = len(unique_Y_train_classes)

# Parameters - Modifiable for Various Configurations
embedding_size = 120
batch_size = 100
no_epochs = 5 # One iteration over all of the training data
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

def model_base(input_shape, network, include_top=False, pooling=None):
    input = Input(shape=input_shape, name='input')
    encoded = network(input)
    prediction = Dense(nb_classes, activation='softmax', bias_initializer=initialize_bias)(encoded)
    output = Model(inputs=[input],outputs=[prediction])
    return output

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


def train_model_base():
    build_representation = embedding_network(num_frames, h, w, d, dimensions=embedding_size)
    cc_model = model_base(input_shape=(num_frames, h, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    cc_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, 
                    metrics=[metrics.MeanSquaredError(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                    tf.keras.metrics.CategoricalAccuracy(name='acc'),
                    CategoricalTruePositives(nb_classes, batch_size)])
    build_representation.summary()
    cc_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, 'CC_Model.png', show_shapes=True)

    labels_train = K.one_hot(Y_train_numeric, nb_classes)
    if reason != 6 or reason != 7:
        trained_history = cc_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None, 
                                validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
        # TO : YOU HAVE TO COMPLETE THIS CORRECTLY
        trained_history = cc_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None, 
                                validation_split=None, validation_data=X_test, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    cc_model.save(paths.model_save + model_name + '_CC' + str(embedding_size))
    with open(paths.history_save + model_name + 'CC' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)

def train_triplet_model():
    build_representation = embedding_network(num_frames, 30, w, d, dimensions=embedding_size)
    tri_model = triplets_model(input_shape=(num_frames, 30, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    tri_model.compile(loss=[None],optimizer=optimizer,sample_weight_mode="temporal")
    build_representation.summary()
    tri_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, 'Triplet_Model.png', show_shapes=True)

    labels_train = K.one_hot(Y_train_numeric, nb_classes)
    if reason != 6 or reason != 7:
        trained_history = tri_model.fit(x=gd.triplet_generator(batch_size, X_train, Y_train), y=None, batch_size=batch_size, epochs=no_epochs, 
                                    verbose=1, callbacks=None, validation_data=None, shuffle=True, class_weight=None, 
                                    sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
         # TO : YOU HAVE TO COMPLETE THIS CORRECTLY
        trained_history = tri_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None, 
                                validation_split=None, validation_data=X_test, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    tri_model.save(paths.model_save + model_name + '_Triplet' + str(embedding_size))
    with open(paths.history_save + model_name + '_Triplet' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)


def train_quadruplet_model():
    build_representation = embedding_network(num_frames, 30, w, d, dimensions=embedding_size)
    quad_model = quadruplet_model(input_shape=(num_frames, 30, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    quad_model.compile(loss=[None],optimizer=optimizer,sample_weight_mode="temporal")
    build_representation.summary()
    quad_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, 'Quadruplet_Model.png', show_shapes=True)

    labels_train = K.one_hot(Y_train_numeric, nb_classes)
    if reason != 6 or reason != 7:
        trained_history = quad_model.fit(x=gd.quadruplet_generator(batch_size, X_train, Y_train), y=None, batch_size=batch_size, epochs=no_epochs, 
                                    verbose=1, callbacks=None, validation_data=None, shuffle=True, class_weight=None, 
                                    sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)
    else:
         # TO : YOU HAVE TO COMPLETE THIS CORRECTLY
        trained_history = quad_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None, 
                                validation_split=None, validation_data=X_test, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    quad_model.save(paths.model_save + model_name + '_Quadruplet' + str(embedding_size))
    with open(paths.history_save + model_name + 'Quadruplet' + str(embedding_size), 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)



train_model_base()

