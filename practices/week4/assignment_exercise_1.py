import numpy as np
import matplotlib.pyplot as plt
import pickle


# Load the training and test data from the Pickle file
with open("../datasets/mnist_dataset_unscaled.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data_scaled = (train_data - pixel_mean) / pixel_std
test_data_scaled = (test_data - pixel_mean) / pixel_std


# Plot a histogram of pixel values
hist, bins = np.histogram(train_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset scaled")
plt.show()


hist, bins = np.histogram(test_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Test dataset scaled")
plt.show()


for idx in range(4):
  image = train_data[idx].reshape(28,28)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()