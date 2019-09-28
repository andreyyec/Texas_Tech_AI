import numpy as np
import pandas as pd
import pickle
import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt












"""
### PRACTICE #1


"""



"""
### PRACTICE #2

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

"""



"""
### PRACTICE #3

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

# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.show()


# Plot a ROC curve (Receiver Operating Characteristic)
# Compares true positive rate with false positive rate
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))


# Predict new labels for training data
Y_pred_proba_training = model.predict(train_data[cols])
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))

"""



"""
### PRACTICE #4

# Load the training and test data from the Pickle file
with open("../datasets/credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a Decision Tree classifier
model = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=350)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))

# Extract the feature importances
importances = model.feature_importances_

# Now make a list and sort it
feature_importance_list = []
for idx in range(len(importances)):
    feature_importance_list.append( (importances[idx], cols[idx]) )

feature_importance_list.sort(reverse=True)

for importance,feature in feature_importance_list:
    print("Feature: {:22s} Importance: {:.4f}".format(feature, importance))
"""



"""
### PRACTICE #5

# Load the training and test data from the Pickle file
with open("../datasets/credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Select columns of interest
cols = train_data.columns

# Create and train a Random Forest classifier
model = sklearn.ensemble.RandomForestClassifier(\
    n_estimators=100,
    min_samples_leaf=20)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get prediction probabilities
Y_pred_proba = model.predict_proba(test_data[cols])[::,1]

auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("Test AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
Y_pred_proba_training = model.predict_proba(train_data[cols])[::,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))

# Extract the feature importances
importances = model.feature_importances_

# Now make a list and sort it
feature_importance_list = []
for idx in range(len(importances)):
    feature_importance_list.append( (importances[idx], cols[idx]) )

feature_importance_list.sort(reverse=True)

for importance,feature in feature_importance_list:
    print("Feature: {:22s} Importance: {:.4f}".format(feature, importance))

# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
# Compares true positive rate with false positive rate
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
"""

