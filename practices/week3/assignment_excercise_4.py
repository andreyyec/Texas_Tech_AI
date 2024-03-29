import pickle
import sklearn.tree
import sklearn.metrics


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
