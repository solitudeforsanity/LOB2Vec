import sys
sys.path.append("/rds/general/user/kk2219/home/LOB2Vec/src")

import config
import os
import paths
import numpy as np
np.set_printoptions(threshold=25)
import tensorflow as tf
import data_preparation.data_cleansing as dc
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Input, Model, metrics, backend as K

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, Y_values, X_values, gen_type, n_classes, shuffle=False):
        'Initialization'
        self.dim = (config.num_frames, config.h, config.w, config.d)
        self.batch_size = config.batch_size
        self.Y_values = Y_values
        self.X_values = X_values
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.gen_type = gen_type

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_values) / self.batch_size))

    def __getitem__(self, index):
        """
        gen_type 0: Single task learning
        gen_type 1: Multi task learning
        gen_type 2: Triplets
        gen_type 3: Quandrupulets
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if self.gen_type == 0:
            X, Y = self.__gen_single_task(indexes)
        elif self.gen_type == 1:
            X, Y = self.__gen_multi_task(indexes)
        elif self.gen_type == 2:
            X, Y = self.__get_triplet()
        elif self.gen_type == 3:
            X, Y = self.__gen_single_task(indexes)
        return X, Y

    def __gen_single_task(self, index):
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))
        y_batch = np.zeros((self.batch_size))
        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
       # print('Getting Single Task')
        #print(y_batch)
       # print(self.Y_values[0])
       # print(self.Y_values[1])
        #print(self.Y_values[2])
       # print(index)
        return np.stack(x_batch, 0), K.one_hot(y_batch, self.n_classes)


    def __gen_multi_task(self, index):
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))
        y_batch = np.zeros((self.batch_size, len(self.Y_values[0])))
        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
        return np.stack(x_batch, 0), {"side": y_batch[:,0].astype(int), "action": y_batch[:,1].astype(int), "price_level": y_batch[:,2].astype(int), "liquidity": y_batch[:,3].astype(int)}

    def old__gen_multi_task(self, index):
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))
        y_batch = np.zeros((self.batch_size, len(self.Y_values[0])))
        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
        return np.stack(x_batch, 0), {"side": K.one_hot(y_batch[:,0], config.side_class_size), "action": K.one_hot(y_batch[:,1], config.action_class_size), "price_level": K.one_hot(y_batch[:,2], config.price_level_class_size), "liquidity": K.one_hot(y_batch[:,3], 39)}


    def __get_triplet(self):
        triplets = [np.zeros((config.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3])) for i in range(3)]

       # for i in range(self.batch_size):
       #    idx_a = np.random.choice(self.labels.shape[0], 1, replace=False)
       #    idx_p = np.random.choice([i for i, v in enumerate(self.labels) if v == self.labels[idx_a]], 1, replace=False)
       #    idx_n = np.random.choice([i for i, v in enumerate(self.labels) if v != self.labels[idx_a]], 1, replace=False)
       #    triplets[0][i,:,:,:,:] = np.load(self.path_to_X + str(config.num_frames) + '_X/' + str(idx_a[0]) + '.npy')
       #    triplets[1][i,:,:,:,:] = np.load(self.path_to_X + str(config.num_frames) + '_X/' + str(idx_p[0]) + '.npy')
       #    triplets[2][i,:,:,:,:] = np.load(self.path_to_X + str(config.num_frames) + '_X/' + str(idx_n[0]) + '.npy')

       #    Y = np.load(self.path_to_X + str(num_frames) + '_Z.npy')
       #    y = Y[idx_a[0]]
       # return [triplets[0], triplets[1], triplets[2]], K.one_hot(y, self.n_classes)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_values))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_generator_data(stock, reason, gen_type):
    if reason == 0:
        # Data for development.
        train_path = paths.source_train_dev
        val_path = paths.source_val_dev
        test_path = paths.source_test_dev
        model_name = 'Dev_'
    elif reason == 1:
       # Training
        train_path = paths.source_train
        val_path = paths.source_val
        test_path = paths.source_test
        model_name = 'Main_'

    X_train, Y_train, Z_train = dc.convert_data_to_labels(stock, train_path)
    X_test, Y_test, Z_test = dc.convert_data_to_labels(stock, test_path)
    X_val, Y_val, Z_val = dc.convert_data_to_labels(stock, val_path)

    #np.savetxt('Y_train_one_hot.csv', K.one_hot(Y_train, 39), delimiter=',')
    
    #np.savetxt('Y_train.csv', Y_train, delimiter=',')
 
    """
    result1 = np.where(Z_train == 1)
    result2 = np.where(Z_test == 1)
    result3 = np.where(Z_val == 1)

    X_train, Y_train, Z_train = X_train[result1], Y_train[result1], Z_train[result1]
    X_test, Y_test, Z_test = X_test[result2], Y_test[result2], Z_test[result2]
    X_val, Y_val, Z_val = X_val[result3], Y_val[result3], Z_val[result3]
    """

    if gen_type == 0:
        validation_generator = DataGenerator(Z_val, X_val, gen_type, config.nb_st_classes)
        training_generator = DataGenerator(Z_train, X_train, gen_type, config.nb_st_classes)
        test_generator = DataGenerator(Z_test, X_test, gen_type, config.nb_st_classes)
    elif gen_type == 1:
        validation_generator = DataGenerator(Y_val, X_val, gen_type, 39)
        training_generator = DataGenerator(Y_train, X_train, gen_type, 39)
        test_generator = DataGenerator(Y_test, X_test, gen_type, 39)
      
    return training_generator, validation_generator, test_generator, model_name, len(X_train), len(X_val)

def tests():
    import numpy as np
    testg, m, x, t, h, hh = get_generator_data('USM_NASDAQ.npy', 0, 1) 
   # print(testg.__getitem__(4)[0])
    tfliquidity = np.array(testg.__getitem__(0)[1]['liquidity'])
    tfside = np.array(testg.__getitem__(0)[1]['side'])
    tfprice_level = np.array(testg.__getitem__(0)[1]['price_level'])
    tfaction = np.array(testg.__getitem__(0)[1]['action'])
    print('My liqdui')
    print(tfliquidity)
    np.savetxt('tfliquidity.csv', tfliquidity, delimiter=',')
    np.savetxt('tfside.csv', tfside, delimiter=',')
    np.savetxt('tfprice_level.csv', tfprice_level, delimiter=',')
    np.savetxt('tfaction.csv', tfaction, delimiter=',')

    y_labels = testg.__getitem__(0)[1]['liquidity']
    #print(tf.argmax(y_labels, axis=1))
    #print(np.array(testg.__getitem__(0)[0]).shape)
   # print(testg.__getitem__(0)[1])
    #np.savetxt("test.csv", np.array(testg.__getitem__(0)[1]), delimiter=",")
    print('HI N--------------------------------')

    def change_to_right(wrong_labels):
        right_labels=[]
        for x in wrong_labels:
            for i in range(0,len(wrong_labels[0])):
                if x[i]==1:
                    right_labels.append(i+1)
        return right_labels

    wrong_labels = np.array([[0,0,1,0], [0,0,1,0], [1,0,0,0],[0,1,0,0]])
    right_labels = tf.convert_to_tensor(np.array(change_to_right(wrong_labels)))
    print(wrong_labels.shape)
    print(right_labels.shape)
    print(right_labels)

    from sklearn.preprocessing import OneHotEncoder

    

#tests()

