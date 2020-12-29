import sys
sys.path.append('.')

import numpy as np
import paths
import pickle
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from loss_and_metrics import CategoricalTruePositives
from tensorflow.keras import backend as K

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
    sns.lineplot(range(1, epoch+1), History.history['action_loss'], label='Train Price Level loss')
    sns.lineplot(range(1, epoch+1), History.history['val_action_loss'], label='Validation Price Level loss')
    sns.lineplot(range(1, epoch+1), History.history['action_loss'], label='Train Liquidity Level loss')
    sns.lineplot(range(1, epoch+1), History.history['val_action_loss'], label='Validation Liquidity Level loss')
    plt.legend(['side:train', 'side:val', 'action:train', 'action:val', 'price-level:train', 'price-level:val', 'liquidity-level:train', 'liquidity-level:val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Loss Graph")
    plt.savefig(paths.result_images + name + '_loss_multi.png')

def multi_acc(History, epoch, name):
    plt.figure(figsize=(20,10))
    sns.lineplot(range(1, epoch+1), History.history['side_binary_accuracy'], label='Train Side Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_side_binary_accuracy'], label='Validation Side Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['action_categorical_accuracy'], label='Train Action Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_action_categorical_accuracy'], label='Validation Action Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['action_categorical_accuracy'], label='Train Price Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_action_categorical_accuracy'], label='Validation Price Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['action_categorical_accuracy'], label='Train Liquidity Level Accuracy')
    sns.lineplot(range(1, epoch+1), History.history['val_action_categorical_accuracy'], label='Validation Liquidity Level Accuracy')
    plt.legend(['side:train', 'side:val', 'action:train', 'action:val', 'price-level:train', 'price-level:val', 'liquidity-level:train', 'liquidity-level:val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(paths.result_images + name + '_acc_multi.png')

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(25,25))
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
