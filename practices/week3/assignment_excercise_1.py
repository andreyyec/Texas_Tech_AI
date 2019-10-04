import numpy as np
import pickle
import sklearn.linear_model
import sklearn.metrics

# Load the training and test data from the Pickle file
with open("../datasets/credit_card_default_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)


# Select columns of interest
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print the coefficients
#print(model.intercept_)
#print(model.coef_)

# Predict new labels for test data
Y_pred = model.predict(test_data[cols])

for idx in range(10):
    print("Predicted: {:6.3f} Correct: {:6d}"\
          .format(float(Y_pred[idx]), test_labels.values[idx]))