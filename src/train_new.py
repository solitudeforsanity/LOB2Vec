import generate_data as gd
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import CategoricalTruePositives
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

# Get Data and set classes 
X_train, Y_train, X_test, Y_test, model_name = gd.get_data(1) 
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
embedding_size = 100
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
    out = TCN(nb_filters=128, kernel_size=2, return_sequences=False, dilations=[1, 2, 4, 8, 16, 32, 64], 
              activation=tf.keras.activations.swish, nb_stacks=1, dropout_rate=0.08)(out)
    out = Flatten()(out)
    out = Dense(embedding_size, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(out)    
    return Model(inputs=inp, outputs=out)

def model_base(input_shape, embedding, include_top=False, pooling=None):
    input = Input(shape=input_shape, name='input')
    encoded = embedding(input)
    prediction = Dense(nb_classes, activation='softmax',bias_initializer=initialize_bias)(encoded)
    output = Model(inputs=[input],outputs=[prediction])
    return output

def train_model_base():
    build_representation = embedding_network(num_frames, h, w, d, dimensions=embedding_size)
    cc_model = model_base(input_shape=(num_frames, h, w, d), embedding=build_representation)
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
    trained_history = cc_model.fit(x=X_train, y=labels_train, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None, 
                                validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, 
                                sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None)

    cc_model.save(paths.model_save + model_name)
    with open(paths.history_save + model_name, 'wb') as file_pi:
        pickle.dump(trained_history.history, file_pi)

train_model_base()

