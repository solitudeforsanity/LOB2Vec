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
    def __init__(self, Y_values, X_values, S_values, M_values, Y_lob_price, gen_type, n_classes, shuffle=False):
        'Initialization'
        self.dim = (config.num_frames, config.h, config.w, config.d)
        self.batch_size = config.batch_size
        self.Y_values = Y_values
        self.S_values = S_values
        self.M_values = M_values
        self.X_values = X_values
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.gen_type = gen_type
        self.Y_lob_price = Y_lob_price

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
        y_lob_price = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.dim[3]))

        for idx_arr, idx  in enumerate(index):
            x_batch[idx_arr] = self.X_values[idx]
            s_batch[idx_arr] = self.S_values[idx]
            m_batch[idx_arr] = self.M_values[idx]
            y_batch[idx_arr] = self.Y_values[idx]
            y_lob_price[idx_arr] = self.Y_lob_price[idx]
       # Use this for COnv or all predictions
      #  return [np.stack(m_batch, 0), np.stack(s_batch, 0), np.stack(x_batch, 0)], {"mid": y_batch[:,7].astype(float), "price": y_batch[:,4].astype(float), "side": y_batch[:,0].astype(int),\
       #                                                                            "action": y_batch[:,1].astype(int), "price_level": y_batch[:,2].astype(int), "liquidity": y_batch[:,3].astype(int), "y_lob_price" : y_lob_price}
        return [np.stack(y_lob_price, 0)], {"y_lob_prices" : y_lob_price}

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

    x_lob_states_train, x_lob_price_train, x_lob_qty_train, x_mid_price_train, x_spread_train, y_df_shifted_train,\
                    z_df_shifted_train, y_lob_price_train, robust_scaler_train = dc.convert_data_to_labels(stock, train_path, robust_scaler)
    x_lob_states_test, x_lob_price_test, x_lob_qty_test, x_mid_price_test, x_spread_test, y_df_shifted_test,\
                    z_df_shifted_test, y_lob_price_test, robust_scaler_test = dc.convert_data_to_labels(stock, test_path, robust_scaler)
    x_lob_states_val, x_lob_price_val, x_lob_qty_val, x_mid_price_val, x_spread_val, y_df_shifted_val, \
                    z_df_shifted_val, y_lob_price_val, robust_scaler_val = dc.convert_data_to_labels(stock, val_path, robust_scaler)



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
        # Use below and change clasess as this is wrong
        training_generator = DataGenerator(y_df_shifted_train, x_lob_price_train, x_spread_train, x_mid_price_train, y_lob_price_train, gen_type, 39)
        test_generator = DataGenerator(y_df_shifted_test, x_lob_price_test, x_spread_test, x_mid_price_test, y_lob_price_test, gen_type, 39)
        validation_generator = DataGenerator(y_df_shifted_val, x_lob_price_val, x_spread_val, x_mid_price_val, x_lob_states_val, gen_type, 39)
    elif gen_type == 1:
        training_generator = DataGenerator(y_df_shifted_train, x_lob_price_train, x_spread_train, x_mid_price_train, y_lob_price_train, gen_type, 39)
        test_generator = DataGenerator(y_df_shifted_test, x_lob_price_test, x_spread_test, x_mid_price_test, y_lob_price_test, gen_type, 39)
        validation_generator = DataGenerator(y_df_shifted_val, x_lob_price_val, x_spread_val, x_mid_price_val, x_lob_states_val, gen_type, 39)
        
       
    return training_generator, validation_generator, test_generator, model_name, len(x_lob_states_train), len(x_lob_states_val), robust_scaler

def tests():
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
    robust_scaler = RobustScaler()
    train_g, val_g, test_g, name, len_x, len_val, scaler = get_generator_data('USM_NASDAQ.npy', 0, 1, robust_scaler) 
    # Parameters for Testing - Amend to these Values in Config. 
    batch_size = 23422
    no_epochs = 1
    num_frames = 1
  
    print(test_g.__len__())
    tfmid = np.array(test_g.__getitem__(0)[1]['mid'])
    tfprice = np.array(test_g.__getitem__(0)[1]['price'])
    tfliquidity = np.array(test_g.__getitem__(0)[1]['liquidity'])
    tfside = np.array(test_g.__getitem__(0)[1]['side'])
    tfprice_level = np.array(test_g.__getitem__(0)[1]['price_level'])
    tfaction = np.array(test_g.__getitem__(0)[1]['action'])
    
    np.savetxt('tfmid.csv', tfmid, delimiter=',')
    np.savetxt('tfprice.csv', tfprice, delimiter=',')
    np.savetxt('tfliquidity.csv', tfliquidity, delimiter=',')
    np.savetxt('tfside.csv', tfside, delimiter=',')
    np.savetxt('tfprice_level.csv', tfprice_level, delimiter=',')
    np.savetxt('tfaction.csv', tfaction, delimiter=',')