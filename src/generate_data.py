import numpy as np
import paths
from tensorflow.keras.utils import Sequence

num_frames = 500
t, h, w, d = num_frames, 30, 2, 2

def get_triplet_batch(batch_size, lob_states, labels):
    n_examples, t, h, w, d = lob_states.shape
    triplets = [np.zeros((batch_size, t, h, w, d)) for i in range(3)]

    for i in range(batch_size):
        idx_a = np.random.choice(labels.shape[0], 1, replace=False)
        idx_p = np.random.choice([i for i, v in enumerate(labels) if v == labels[idx_a]], 1, replace=False)
        idx_n = np.random.choice([i for i, v in enumerate(labels) if v != labels[idx_a]], 1, replace=False)
        
        triplets[0][i,:,:,:,:] = lob_states[idx_a]
        triplets[1][i,:,:,:,:] = lob_states[idx_p]
        triplets[2][i,:,:,:,:] = lob_states[idx_n]
        
    return [triplets[0], triplets[1], triplets[2]]

def triplet_generator(batch_size, X_train, Y_train):
    while True:
        triplets = get_triplet_batch(batch_size, X_train, Y_train)
        yield triplets

def get_quadruplets_batch(batch_size, lob_states, labels):
    n_examples, t, h, w, d = lob_states.shape
    quadruplets = [np.zeros((batch_size, t, h, w, d)) for i in range(4)]

    for i in range(batch_size):
        idx_a = np.random.choice(labels.shape[0], 1, replace=False)
        idx_p , idx_n, idx_n2 = 1, 2, 3
      #  idx_p = np.random.choice([i for i, v in enumerate(labels) if v == labels[idx_a]], 1, replace=False)
       # idx_n = np.random.choice([i for i, v in enumerate(labels) if v != labels[idx_a]], 1, replace=False)
       # idx_n2 = np.random.choice([i for i, v in enumerate(labels) if v != labels[idx_a]], 1, replace=False)
        
        quadruplets[0][i,:,:,:,:] = lob_states[idx_a]
        quadruplets[1][i,:,:,:,:] = lob_states[idx_p]
        quadruplets[2][i,:,:,:,:] = lob_states[idx_n]
        quadruplets[3][i,:,:,:,:] = lob_states[idx_n2]
        
    return [quadruplets[0], quadruplets[1], quadruplets[2], quadruplets[3]]

def quadruplet_generator(batch_size, X_train, Y_train):
    while True:
        quadruplets = get_quadruplets_batch(batch_size, X_train, Y_train)
        yield quadruplets

def get_data(reason):
    """ Retrieves data based on needs of the training job. Some we split data into train and test, where as others we allow keras to split it automatically. 
    Args:
        reason (int): 
        0 : Testing for working code. We name this 'Test_Model'
        1 : Train on all data with a 0.2 validation set
        2 : Train on data subset with validation in the early period with 0.2 in keras
        3 : Train on data subset with validation in the later period with 0.2 in keras
        4 : Train on early period, validation on late period
        5 : Train on late period, validation on early period
    Returns:
        list : Returns a list of arrays for X_train / X_val and Y_train / Y_val
    """

    if reason == 0:
        X_train = np.load(paths.dev_dest + '/' + str(num_frames) + '_X.npy')
        Y_train = np.load(paths.dev_dest + '/' + str(num_frames) + '_Y.npy')
        X_val = np.load(paths.dev_dest + '/' + str(num_frames) + '_X.npy')
        Y_val = np.load(paths.dev_dest + '/' + str(num_frames) + '_Y.npy')
        model_name = 'Model_Dev'
    elif reason == 1:
        X_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_X.npy')
        Y_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Y.npy')
        X_train = X_2016
        Y_train = Y_2016
        X_val = []
        Y_val = []
        model_name = 'Model_2016'
    elif reason == 2:
        X_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_X.npy')
        Y_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Y.npy')
        X_train = X_2017
        Y_train = Y_2017
        X_val = []
        Y_val = []
        model_name = 'Model_2017'
    elif reason == 3:
        # For Validation data only 2016
        X_train = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Test2016_X.npy')
        Y_train = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Test2016_Y.npy')
        X_val = []
        Y_val = []
        model_name = 'ModelTest_2016'
    elif reason == 4 :
        # For Validation data only 2017
        X_train = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Test2017_X.npy')
        Y_train = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Test2017_Y.npy')
        X_val = []
        Y_val = []
        model_name = 'ModelTest_2017'
    else:
        X_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_X.npy')
        Y_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Y.npy')
        X_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_X.npy')
        Y_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Y.npy')
        if reason == 5:
            X_train = X_2016.append(X_2017)
            Y_train = Y_2016.append(Y_2017)
            X_val = []
            Y_val = []
            model_name = 'Model_All'
        elif reason == 6:
            X_train = X_2016
            Y_train = Y_2016
            X_val = X_2017
            Y_val = Y_2017
            model_name = 'Model_OldNew'
        elif reason == 7:
            X_train = X_2017
            Y_train = Y_2017
            X_val = X_2016
            Y_val = Y_2016
            model_name = 'Model_NewOld'
        
    return X_train, Y_train, X_val, Y_val, model_name

    # -------------- FUTURE WORK ---------------- #
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(t,h,w,d), n_classes=25, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = [np.zeros((self.batch_size, t, h, w, d)) for i in range(1)]
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(paths.dev_dest_generator + 'X/' + str(2) + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, K.one_hot(y, self.n_classes)

def generate_data_using_generator():
    labels = y = np.load(paths.dev_dest_generator + 'Y.npy')
    x_ids = list(range(0, len(labels)))
    training_generator = DataGenerator(x_ids, labels)
    return training_generator