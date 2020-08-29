from collections import defaultdict
import pandas as pd
import numpy as np
import paths
import model
import generate_data as gd
from sklearn.preprocessing import normalize
from scipy.stats import logistic
from os.path import join
from tqdm import tqdm
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


import warnings
for i in [DeprecationWarning,FutureWarning,UserWarning]:
    warnings.filterwarnings("ignore", category = i)

path_model = join(paths.model_save,'MyModel.hdf5')

def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)

def check_loss():
    batch_size = 10
    shape = (batch_size, 4096)

    p1 = normalize(np.random.random(shape))
    n = normalize(np.random.random(shape))
    p2 = normalize(np.random.random(shape))
    
    input_tensor = [K.variable(p1), K.variable(n), K.variable(p2)]
    out1 = K.eval(triplet_loss(input_tensor))
    input_np = [p1, n, p2]
    out2 = triplet_loss_np(input_np)

    assert out1.shape == out2.shape
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))

def triplets_model(input_shape, network, include_top=False, pooling=None):

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    # Get the embedded values
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [encoded_a, encoded_p, encoded_n]
       
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))

    return triplet_model
    

def train_model_triplet(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, model_name, nb_classes, reason):
    build_representation = model.embedding_network(num_frames, 30, w, d, embedding_size)
    tri_model = triplets_model(input_shape=(num_frames, 30, w, d), network=build_representation)

    csv_logger = CSVLogger(paths.logs_save + model_name + str(reason) + '_CC_Representation' + str(embedding_size) + '.log', append=True, separator=';')
    checkpoint = ModelCheckpoint(path_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    callbacks_list = [checkpoint, early_stopper, csv_logger]  # early

    tri_model.compile(loss=None, optimizer=Adam(0.01))

    trained_history = tri_model.fit(x=gd.triplet_generator(batch_size, X_train, Y_train), y=None, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=callbacks_list, 
                                    validation_split=None, validation_data=gd.triplet_generator(batch_size, X_val, Y_val), shuffle=True, class_weight=None, 
                                    sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled, validation_steps=None, use_multiprocessing=False)
    eva_plot(trained_history, 4)

def eva_plot(History, epoch):
    plt.figure(figsize=(20,10))
#     sns.lineplot(range(1, epoch+1), History.history['acc'], label='Train Accuracy')
#     sns.lineplot(range(1, epoch+1), History.history['val_acc'], label='Test Accuracy')
#     plt.legend(['train', 'validaiton'], loc='upper left')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.show()
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['loss'], label='Train loss')
    sns.lineplot(range(1, epoch+1), History.history['val_loss'], label='Test loss')
    plt.legend(['train', 'validaiton'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Loss Graph")
    plt.savefig('loss_tr.png')
    


if __name__ == "__main__":
    my_reason = 0
    X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, model_name, nb_classes = model.return_parameters(my_reason)
    train_model_triplet(X_train, Y_train, X_val, Y_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, model_name, nb_classes, my_reason)