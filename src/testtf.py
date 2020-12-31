import tensorflow as tf
import tensorflow.keras.losses as ls

true = [0.0, 1.0]
pred = [[0.1,0.9],[0.0,1.0]]

tt = tf.convert_to_tensor(true)
tp = tf.convert_to_tensor(pred)

l = ls.SparseCategoricalCrossentropy()
ret = l(tt,tp)

print(ret)  