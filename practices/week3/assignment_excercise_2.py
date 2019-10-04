import numpy as np
import pickle
import sklearn.linear_model

# Load the training and test data from the Pickle file
with open("../datasets/credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Predict new labels for test data
Y_pred_proba = model.predict(test_data[cols])


for value in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:

    # Binarize the predictions by comparing to a threshold
    threshold = value
    print("Threshold: ", threshold)
    Y_pred = (Y_pred_proba > threshold).astype(np.int_)

    # Count how many are predicted as 0 and 1
    print("Predicted as 1: ", np.count_nonzero(Y_pred))
    print("Predicted as 0: ", len(Y_pred) - np.count_nonzero(Y_pred))


    # Compute the statistics
    cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
    print("Confusion Matrix:")
    print(cmatrix)

    accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred)
    print("Accuracy: {:.3f}".format(accuracy))

    precision = sklearn.metrics.precision_score(test_labels, Y_pred)
    print("Precision: {:.3f}".format(precision))

    recall = sklearn.metrics.recall_score(test_labels, Y_pred)
    print("Recall: {:.3f}".format(recall))