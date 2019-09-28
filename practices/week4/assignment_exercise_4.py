import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.ensemble

# Load the training and test data from the Pickle file
with open("../datasets/mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

num_classes = len(np.unique(train_labels))

# Train a Decision Tree classifier
model = sklearn.ensemble.RandomForestClassifier(\
    n_estimators = 100,
    min_samples_leaf = 1)
model.fit(train_data, train_labels)

# Predict the labels for all the test cases
Y_pred = model.predict(test_data)

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))

# Compute the prediction accuracy against the training data
print("Against training set:")
Y_pred_training = model.predict(train_data)
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

i = 0
matches = 0

while i < len(Y_pred) and matches < 10:
    print("----- Loop -----")
    print("i: " + str(i))
    print("matches: " + str(matches))

    c_label = test_labels[i]
    c_guess = Y_pred[i]

    if c_label != c_guess:
        image = test_data[i].reshape(28, 28)
        plt.figure()
        plt.imshow(image, cmap="gray_r")
        plt.title("Actual Value: " + str(c_label) + ", Guessed Value: " + str(c_guess))
        plt.show()

        matches += 1

    i += 1