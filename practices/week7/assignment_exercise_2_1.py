import pickle
import numpy as np
import sklearn.metrics
import keras.models
import keras.layers
import keras.utils
import keras.regularizers
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("../datasets/mnist_dataset_unscaled.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)
num_classes = test_labels_onehot.shape[1]

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes_l1 = 100
n_nodes_l2 = 200
batch_size = 32
n_epochs = 100
regularization_scale = 0

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num nodes in L1: {} L2: {}".format(n_nodes_l1, n_nodes_l2))
print("Regularization scale: {}".format(regularization_scale))

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# Hidden layer 1
model.add(keras.layers.Dense(n_nodes_l1, input_shape=(n_inputs,),
                             activation='elu',
                             kernel_initializer='he_normal', bias_initializer='zeros',
                             kernel_regularizer=keras.regularizers.l2(regularization_scale)))

model.add(keras.layers.Dense(n_nodes_l2,
                             activation='elu',
                             kernel_initializer='he_normal', bias_initializer='zeros',
                             kernel_regularizer=keras.regularizers.l2(regularization_scale)))

# Output layer
model.add(keras.layers.Dense(num_classes,
                             activation='softmax',
                             kernel_initializer='glorot_normal', bias_initializer='zeros',
                             kernel_regularizer=keras.regularizers.l2(regularization_scale)))

# Define the optimizer
# optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.Adadelta(lr=1.0)
# optimizer = keras.optimizers.Adagrad(lr=0.01)

# Define cost function and optimization strategy
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the neural network
history = model.fit(
    train_data,
    train_labels_onehot,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_data=(test_data, test_labels_onehot),
    verbose=2,
)

# Find the best costs & metrics
test_accuracy_hist = history.history['val_acc']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['acc']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

# Plot the history of the cost
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()