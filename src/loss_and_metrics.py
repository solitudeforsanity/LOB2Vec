import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class QuadrupletLossLayer(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.debugeric = 1
        super(QuadrupletLossLayer, self).__init__(**kwargs)
    
    def quadruplet_loss(self, inputs):
        anchor, positive, negative, negative2 = inputs
        ap_dist = K.sum(K.square(anchor-positive), axis=-1)
        an_dist = K.sum(K.square(anchor-negative), axis=-1)
        nn_dist = K.sum(K.square(negative-negative2), axis=-1)
        
        #square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)
        
        return K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)
    
    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss

class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))
        self.cat_true_positives.assign_add(true_poss)

    def result(self):
        return self.cat_true_positives

