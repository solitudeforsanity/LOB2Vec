import numpy as np
import paths

num_frames = 500

def get_triplet_batch_spoof(batch_size, lob_states, labels):
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

def triplet_generator_spoof(batch_size, X_train, Y_train):
    while True:
        triplets = get_triplet_batch_spoof(batch_size, X_train, Y_train)
        yield triplets

def get_data(reason):
    """ Retrieves data based on needs of the training job. Some we split data into train and test, where as others we allow keras to split it automatically. 
    Args:
        reason (int): 
        0 : Testing for working code
        1 : Train on all data with a 0.2 validation set
        2 : Train on data subset with validation in the early period with 0.2 in keras
        3 : Train on data subset with validation in the later period with 0.2 in keras
        4 : Train on early period, validation on late period
        5 : Train on late period, validation on early period
    Returns:
        list : Returns a list of arrays for X_train / X_test and Y_train / Y_test
    """

    if reason == 0:
        X_train = np.load(paths.test_dest + '/' + str(num_frames) + '_small_X.npy')
        Y_train = np.load(paths.test_dest + '/' + str(num_frames) + '_small_Y.npy')
        X_test = np.load(paths.test_dest + '/' + str(num_frames) + '_small_X.npy')
        Y_test = np.load(paths.test_dest + '/' + str(num_frames) + '_small_Y.npy')
    elif reason == 1:
        X_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_X.npy')
        Y_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Y.npy')
        X_train = X_2016
        Y_train = Y_2016
        X_test = []
        Y_test = []
    elif reason == 2:
        X_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_X.npy')
        Y_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Y.npy')
        X_train = X_2017
        Y_train = Y_2017
        X_test = []
        Y_test = []
    else:
        X_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_X.npy')
        Y_2016 = np.load(paths.dest_2016 + '/' + str(num_frames) + '_Y.npy')
        X_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_X.npy')
        Y_2017 = np.load(paths.dest_2017 + '/' + str(num_frames) + '_Y.npy')
        if reason == 3:
            X_train = X_2016.append(X_2017)
            Y_train = Y_2016.append(Y_2017)
            X_test = []
            Y_test = []
        elif reason == 4:
            X_train = X_2016
            Y_train = Y_2016
            X_test = X_2017
            Y_test = Y_2017
        elif reason == 5:
            X_train = X_2017
            Y_train = Y_2017
            X_test = X_2016
            Y_test = Y_2016
        
    return X_train, Y_train, X_test, Y_test