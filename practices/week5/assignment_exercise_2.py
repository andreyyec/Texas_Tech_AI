import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("../datasets/vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
learning_rate = .1
n_iterations = 3000
print_step = 100

# TensorFlow constants

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Initial coefficients & bias

# Start with uniformly random values
W = tf.Variable(tf.random_uniform([ncoeffs, 1], -1.0, 1.0))

# Start the bias with zeros
b = tf.Variable(0.0)

# TensorFlow computations

# Prediction
Y_pred = tf.add(tf.matmul(X, W), b)

# Error
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error))

# Optimize MSE through gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_op = optimizer.minimize(mse)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
print("========  TRAIN DATA: =========")
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, mse_val = sess.run([update_op, mse])
    if (iteration % print_step == 0):
        print("iteration {:4d}:  MSE: {:.1f}".format(iteration, mse_val))
    iteration += 1

# Run a session to retrieve the coefficients & bias
W_result, b_result = sess.run([W, b])

# Print the coefficients
print("Coeffs:", W_result)
print("Bias:", b_result)

# Compute the training RMSE
training_rmse = mse_val ** .5
print("Training RMSE: {:.1f}".format(training_rmse))




X_test = tf.constant(test_data.values.astype(np.float32))
Y_test = tf.constant(test_labels.values.reshape(-1,1).astype(np.float32))
Y_pred_test = tf.add(tf.matmul(X_test, W), b)

# Error
error_test = Y_pred_test - Y_test
mse_test = tf.reduce_mean(tf.square(error_test))

# Optimize MSE through gradient descent
optimizer_test = tf.train.GradientDescentOptimizer(learning_rate)
update_op_test = optimizer_test.minimize(mse_test)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 0
print("========  TEST DATA: =========")
while iteration < n_iterations:
    # Run one iteration of the computation session to update coefficients
    _, mse_test_val = sess.run([update_op_test, mse_test])
    if (iteration % print_step == 0):
        print("iteration {:4d}:  MSE: {:.1f}".format(iteration, mse_test_val))
    iteration += 1

# Compute the training RMSE
test_rmse = mse_test_val ** .5

print("Training RMSE: {:.1f}".format(test_rmse))

