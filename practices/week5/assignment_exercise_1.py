import numpy as np
import pickle
import tensorflow as tf

# Load the training and test data from the Pickle file
with open("../datasets/vehicle_price_dataset_scaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Insert a column of ones
train_data["ones"] = 1

# TensorFlow constants
# Start the bias with zeros
b = tf.Variable(0.0)

# Input vectors
X = tf.constant(train_data.values.astype(np.float32))
Y = tf.constant(train_labels.values.reshape(-1,1).astype(np.float32))

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = tf.transpose(X)
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

# Create TensorFlow session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the session
W_result = sess.run(W)

# Print the coefficients
#print(W_result)

test_data["ones"] = 1

Xtest = tf.constant(test_data.values.astype(np.float32))
Ytest = tf.constant(test_labels.values.reshape(-1,1).astype(np.float32))

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XTtest = tf.transpose(Xtest)
Wtest = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XTtest, Xtest)), XTtest), Ytest)

Y_pred_test = tf.matmul(Xtest, Wtest)

# Create TensorFlow session and initialize it
sess_test = tf.Session()
init_test = tf.global_variables_initializer()
sess.run(init)

# Run the session
W_test_result = sess.run(Y_pred_test)

# Print the coefficients
# print(W_test_result)

for idx in range(10):
    print("Predicted: {:6.3f} Correct: {:6d}"\
          .format(float(W_test_result[idx]), test_labels.values[idx]))
