import sys
sys.path.append('.')
sys.path.append("/rds/general/user/kk2219/home/LOB2Vec/src")

import data_preparation.data_cleansing as dc
import config
import numpy as np
import paths
import pickle
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from loss_and_metrics import CategoricalTruePositives
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix

def loss(History, epoch, name):
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['loss'], label='Train loss')
    sns.lineplot(range(1, epoch+1), History.history['val_loss'], label='Validation loss')
    plt.legend(['train', 'validaiton'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Loss Graph")
    plt.savefig(paths.result_images + name + '_loss.png')

def acc(History, epoch, name):
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['acc'], label='Train Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_acc'], label='Validation Accuracy')

    plt.legend(['train', 'validaiton'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(paths.result_images + name + '_acc.png')

def multi_loss(History, epoch, name):
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['side_loss'], label='Train Side loss')
    sns.lineplot(range(1, epoch+1), History.history['val_side_loss'], label='Validation Side loss')
    sns.lineplot(range(1, epoch+1), History.history['action_loss'], label='Train Action loss')
    sns.lineplot(range(1, epoch+1), History.history['val_action_loss'], label='Validation Action loss')
    sns.lineplot(range(1, epoch+1), History.history['price_level_loss'], label='Train Price Level loss')
    sns.lineplot(range(1, epoch+1), History.history['val_price_level_loss'], label='Validation Price Level loss')
    sns.lineplot(range(1, epoch+1), History.history['liquidity_loss'], label='Train Liquidity Level loss')
    sns.lineplot(range(1, epoch+1), History.history['val_liquidity_loss'], label='Validation Liquidity Level loss')
    plt.legend(['side:train', 'side:val', 'action:train', 'action:val', 'price-level:train', 'price-level:val', 'liquidity-level:train', 'liquidity-level:val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Loss Graph")
    plt.savefig(paths.result_images + name + '_loss_multi.png')

def multi_acc(History, epoch, name):
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['side_sparse_categorical_accuracy'], label='Train Side Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_side_sparse_categorical_accuracy'], label='Validation Side Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['action_sparse_categorical_accuracy'], label='Train Action Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_action_sparse_categorical_accuracy'], label='Validation Action Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['price_level_sparse_categorical_accuracy'], label='Train Price Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_price_level_sparse_categorical_accuracy'], label='Validation Price Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['liquidity_sparse_categorical_accuracy'], label='Train Liquidity Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_liquidity_sparse_categorical_accuracy'], label='Validation Liquidity Level Accuracy')
    plt.legend(['side:train', 'side:val', 'action:train', 'action:val', 'price-level:train', 'price-level:val', 'liquidity-level:train', 'liquidity-level:val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(paths.result_images + name + '_acc_multi.png')    

def generate_metrics(tf_model, testing_gen, multi_task, stock, model_name, scaler):
    mid, price, price_level = tf_model.predict(testing_gen, steps=testing_gen.__len__())
    for feature_name in config.features_classification:
        if multi_task:
        # loss, side_loss, action_loss, price_level_loss, liquidity_loss, side_binary_accuracy, action_categorical_accuracy, price_level_categorical_accuracy, liquidity_categorical_accuracy = tf_model.evaluate(testing_gen, verbose=2)
            
           # print(side)
           # loss, side_loss, action_loss, price_level_loss, liquidity_loss, side_binary_accuracy, action_categorical_accuracy, price_level_categorical_accuracy, liquidity_categorical_accuracy = tf_model.evaluate(testing_gen, verbose=2)
            #with open():
            
           # create_confusion_matrix(side, 'side', testing_gen, config.nb_mt_classes, config.start_side, config.end_side, stock, model_name)
            #create_confusion_matrix(action, 'action', testing_gen, config.nb_mt_classes, config.start_action, config.end_action, stock, model_name)
            create_confusion_matrix(price_level, 'price_level', testing_gen, config.nb_mt_classes, config.start_price_level, config.end_price_level, stock, model_name)
            # create_confusion_matrix(liquidity, 'liquidity', testing_gen, config.nb_mt_classes, config.start_liquidity, config.end_liquidity, stock, model_name)

    for feature_name in config.features_regression:
        create_line_plot(testing_gen, stock, price, 'price', False, scaler)


def create_line_plot(testing_gen, stock, feature, feature_name, should_rescale, scaler):
        y_labels = np.empty((0))
        for i in range(0, testing_gen.__len__()):
            y_labels = np.append(y_labels, testing_gen.__getitem__(i)[1][feature_name].flatten(), axis=0)
        feature = feature.flatten()
        time = range(len(feature))
        feature_len = len(feature)
        print(feature_len)
        print(len(y_labels))

        if should_rescale:
            feature = feature.reshape(-1,1)
            feature = scaler.inverse_transform(feature)
            feature = feature.reshape(feature_len,)

            y_labels = y_labels.reshape(-1,1)
            y_labels = scaler.inverse_transform(y_labels)
            y_labels = y_labels.reshape(feature_len,)

        print(feature)
        print(y_labels)
        
        plt.figure(figsize=(20,10))
        sns.lineplot(y=feature[10:], x=time[10:], label='Prediction')
        sns.lineplot(y=y_labels[10:], x=time[10:], label='Original')

        plt.savefig(paths.result_images + stock + "_" + feature_name + '_pred.png')

def create_confusion_matrix(feature, feature_name, testing_gen, label_size, start_idx, end_idx, stock, model_name):
        predicted_class = tf.argmax(feature, axis=1)
        y_labels = np.empty((0))
        for i in range(0, testing_gen.__len__()):
            y_labels = np.append(y_labels, testing_gen.__getitem__(i)[1][feature_name], axis=0)
        print(feature_name)
        print(feature)
        print(predicted_class)
        print(y_labels)
        print(feature.shape)
        print(predicted_class.shape)
        print(y_labels.shape)
        #true_class = tf.argmax(y_labels, axis=1)
       
        cnf_matrix = confusion_matrix(y_labels, predicted_class, labels=list(range(start_idx, end_idx)))
        plot_confusion_matrix(cnf_matrix, list(range(start_idx, end_idx)), model_name + str(config.num_frames) + '_convlstm_multi_' + feature_name + '_' + stock[:-11])

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, classes, name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)
    plt.figure(figsize=(25,25))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_points = np.arange(len(classes))
    tick_marks = classes
    plt.xticks(tick_points, tick_marks, rotation=45)
    plt.yticks(tick_points, tick_marks)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(paths.result_images + name + '_confusion_multi.png')

def plot_confusion_matrix_test(classes, name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm = np.array([[0.5,0.6,0.8],[0.0,0.7,0.9],[0.1,0.3,0.6]])

    plt.figure(figsize=(20,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(paths.result_images + name + '_confusion.png')

def seaborn_cm():
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)