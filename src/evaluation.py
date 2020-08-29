import generate_data as gd
import numpy as np
import paths
import pickle
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
from loss_and_metrics import CategoricalTruePositives
from tensorflow.keras import backend as K

history = pickle.load(open(paths.history_save + '/Model_20161_Quadruplet_History120',"rb"))
nb_classes = 25
batch_size = 50

# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Acc.png')

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss.png')

