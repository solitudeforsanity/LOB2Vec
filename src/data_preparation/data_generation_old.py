import sys
sys.path.append("/rds/general/user/kk2219/home/LOB2Vec/src")

import config
import os
import paths
import numpy as np
from os.path import basename
from pathlib import Path
np.set_printoptions(threshold=25)
import tensorflow as tf
import data_preparation.data_cleansing as dc
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Input, Model, metrics, backend as K

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, Y_len, gen_type, data_path, n_classes, stock_name, scaler1, shuffle=False):
        'Initialization'
        self.dim = (config.num_frames, config.h, config.w, config.d)
        self.batch_size = config.batch_size
        self.Y_len = Y_len
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.gen_type = gen_type
        self.stock_name = stock_name
        self.data_path = data_path
        self.scaler1 = scaler1
        self.curr_y_len = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.Y_len / self.batch_size))

    def __getitem__(self, index):
        """
        gen_type 0: Single task learning
        gen_type 1: Multi task learning
        gen_type 2: Triplets
        gen_type 3: Quandrupulets
        """
        if index < self.curr_y_len:
            print('First Entry')
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            print('Second Entry')
            for subdir, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file == self.stock_name:
                        data_path = os.path.join(subdir, file)
                        my_path = Path(data_path)
                        date_path = my_path.parent.parent
                        x_path = date_path / 'X' / file
                        z_path = date_path / 'Z' / file
                        XorY = basename(my_path.parent)
                        if XorY == 'Y':
                            npy_y = np.load(data_path, allow_pickle=True)
                            npy_x = np.load(x_path, allow_pickle=True)
                            npy_z = np.load(z_path, allow_pickle=True)

                            x, p, q, y, z, spread, mid, robust_scaler = dc.retrieve_cleansed_data(npy_x, npy_y, npy_z, file, True, False, self.scaler1)

                            self.X_values = p
                            self.Y_values = y
                            self.M_values = mid
                            self.S_values = spread
                            self.curr_y_len = len(y)

                            indexes = self.indexes[300:4*self.batch_size]
        print('First Entry part 2')
        if self.gen_type == 0:
            X, Y = self.__gen_single_task(indexes)
        elif self.gen_type == 1:
            print('First Entry part 3')
            X, Y = self.__gen_multi_task(indexes)
            print('First Entry part 4')
            print(self.curr_y_len)
            print(self.Y_len)
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

    def __gen_multi_task_less_dim(self, index):
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        y_batch = np.zeros((self.batch_size, len(self.Y_values[0])))
        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
        #  return np.stack(x_batch, 0), {"side": y_batch[:,0].astype(int), "action": y_batch[:,1].astype(int), "price_level": y_batch[:,2].astype(int), "liquidity": y_batch[:,3].astype(int)}
        return np.stack(x_batch, 0), {"price_level": y_batch[:,2].astype(int)}

    def __gen_multi_task(self, index):
        x_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))
        s_batch = np.zeros((self.batch_size, self.dim[0], 1))
        m_batch = np.zeros((self.batch_size, self.dim[0], 1))
        y_batch = np.zeros((self.batch_size, len(self.Y_values[0])))

        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            s_batch[idx_arr] = self.S_values[idx]
            m_batch[idx_arr] = self.M_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
        #  return np.stack(x_batch, 0), {"side": y_batch[:,0].astype(int), "action": y_batch[:,1].astype(int), "price_level": y_batch[:,2].astype(int), "liquidity": y_batch[:,3].astype(int)}
        # mid == 7, spread == 6
        print(index)
        return [np.stack(m_batch, 0), np.stack(s_batch, 0), np.stack(x_batch, 0)], {"mid": y_batch[:,7].astype(float), "price": y_batch[:,4].astype(float), "price_level": y_batch[:,2].astype(int)}


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
        self.indexes = np.arange(self.Y_len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_generator_data(stock, reason, gen_type, robust_scaler):
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

    y_len_train = dc.return_y_len(stock, train_path, robust_scaler, Y_value='Y')
    y_len_test = dc.return_y_len(stock, test_path, robust_scaler, Y_value='Y')
    y_len_val = dc.return_y_len(stock, val_path, robust_scaler, Y_value='Y')

    if gen_type == 0:
        # Use below and change clasess as this is wrong
        validation_generator = DataGenerator(y_len_val, gen_type, val_path, 39, stock, robust_scaler)
        training_generator = DataGenerator(y_len_train, gen_type, train_path, 39, stock, robust_scaler)
        test_generator = DataGenerator(y_len_test, gen_type, test_path, 39, stock, robust_scaler)
    elif gen_type == 1:
        validation_generator = DataGenerator(y_len_val, gen_type, val_path, 39, stock, robust_scaler)
        training_generator = DataGenerator(y_len_train, gen_type, train_path, 39, stock, robust_scaler)
        test_generator = DataGenerator(y_len_test, gen_type, test_path, 39, stock, robust_scaler)
    
    print(training_generator[0])
    return training_generator, validation_generator, test_generator, model_name, 5, 6, robust_scaler

def tests():
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
    robust_scaler = RobustScaler()
    testg, m, x, t, h, hh, rs = get_generator_data('USM_NASDAQ.npy', 0, 1, robust_scaler) 
   # print(testg.__getitem__(4)[0])
    print(testg.__getitem__(0)[1]['price'].flatten())
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


