import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

# Load the training and test data from the Pickle file
with open("../datasets/mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')
train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels_onehot = encoder.transform(test_labels.reshape(-1, 1))
num_classes = len(encoder.categories_[0])

# Train a linear regression classifier
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(train_data, train_labels_onehot)

# Predict the probabilities of each class
Y_pred_proba = model.predict(test_data)

# Pick the maximum
Y_pred = np.argmax(Y_pred_proba, axis=1).astype("uint8")

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))

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




