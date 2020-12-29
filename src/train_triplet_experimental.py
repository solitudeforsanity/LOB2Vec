import model2 as model
import generate_data as gd
import evaluation
import math
import numpy as np
import paths
import pickle
import tensorflow as tf
from loss_and_metrics import TripletLossLayer
from tcn import TCN, tcn_full_summary
from tensorflow.keras import Input, Model, metrics, backend as K
from tensorflow.keras.layers import Dense, Conv2D, Layer, Lambda, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


#----------------TRIPLET MODEL-----------------#

def triplets_model(input_shape, network, include_top=False, pooling=None):

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    # Get the embedded values
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    # Connect the inputs with the outputs
    triplet_net = Model(inputs=[anchor_input,positive_input,negative_input],outputs=[loss_layer])
    return triplet_net

def train_triplet_model(training_gen, validation_gen, model_name, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, val_steps_per_epoch_travelled, batch_size, embedding_size, nb_classes, reason):
    build_representation = model.embedding_network(num_frames, h, w, d, embedding_size)
    tri_model = triplets_model(input_shape=(num_frames, 30, w, d), network=build_representation)
    optimizer = Adam(lr = 0.00006)
    tri_model.compile(loss=[None],optimizer=optimizer,sample_weight_mode="temporal")
    build_representation.summary()
    tri_model.summary()
    tcn_full_summary(build_representation, expand_residual_blocks=True)
    plot_model(build_representation, paths.model_images + '/Triplet_Model.png', show_shapes=True)

    trained_history = tri_model.fit(x=training_gen, y=None, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=None,
                                   shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=steps_per_epoch_travelled*0.9, max_queue_size=10, workers=1, use_multiprocessing=False,
                                   validation_steps=val_steps_per_epoch_travelled*0.9,  validation_batch_size=batch_size, validation_freq=1, validation_data=validation_gen, validation_split=0.0)


    build_representation.save(paths.model_save + model_name + str(reason) + '_Triplet_Representation' + str(embedding_size))
    tri_model.save(paths.model_save + model_name + str(reason) + '_Triplet_Model' + str(embedding_size))
   
    evaluation.loss(trained_history, no_epochs, model_name + '_triplet')
    return tri_model, build_representation

def compute_probs(network,X,Y):
        '''
        Input
            network : current NN to compute embeddings
            X : tensor of shape (m,w,h,1) containing pics to evaluate
            Y : tensor of shape (m,) containing true class
            
        Returns
            probs : array of shape (m,m) containing distances
        
        '''
        m = X.shape[0]
        nbevaluation = int(m*(m-1)/2)
        probs = np.zeros((nbevaluation))
        y = np.zeros((nbevaluation))
        
        #Compute all embeddings for all pics with current network
        embeddings = network.predict(X)
        
        size_embedding = embeddings.shape[1]
        
        #For each pics of our dataset
        k = 0
        for i in range(m):
                #Against all other images
                for j in range(i+1,m):
                    #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                    probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
                    if (Y[i]==Y[j]):
                        y[k] = 1
                        #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                    else:
                        y[k] = 0
                        #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                    k += 1
        return probs,y
    

if __name__ == "__main__":
    my_reason = 0
    gen_type = 1
    print('WORKING ON TRAINING TRIPLET')
    
    training_gen, validation_gen, testing_gen, model_name, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, val_steps_per_epoch_travelled, \
                                                                    batch_size, embedding_size, nb_classes = model.return_parameters(my_reason, gen_type)
    print(training_gen)
    triplet_model, network = train_triplet_model(training_gen, validation_gen, model_name, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, val_steps_per_epoch_travelled, batch_size, \
                                        embedding_size, nb_classes, my_reason)

    probs,yprobs = compute_probs(network,testing_gen[0][1],testing_gen[1])
    