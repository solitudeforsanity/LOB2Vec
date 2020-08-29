import model
import numpy as np

X_train, Y_train, X_val, labels_train, labels_val, num_frames, h, w, d, no_epochs, steps_per_epoch_travelled, batch_size, embedding_size, reason, model_name, nb_classes = model.return_parameters()
print(labels_train.shape)

y = np.empty((50, labels_train.shape[1]))
print(y.shape)

y[15] = labels_train[3]
print(y[15])
print(labels_train[3])