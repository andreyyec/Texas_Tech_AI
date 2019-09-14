import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./datasets/craigslistVehicles.csv")

# Insert a column of ones to serve as x0
df["ones"] = 1

# Compute Price vs. Year linear model
X = df[["ones", "odometer"]].values
Y = df["price"].values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y
intercept = W[0]
coeffs = W[1:]

eq_str = "price = {:.0f} + {:.1f} * odometer".format(intercept, coeffs[0])
print(eq_str)

# Plot the scatterplot of price vs. year, with the trendline on top

minval = min(df["odometer"])
maxval = max(df["odometer"])
x_range = range(minval, maxval+1)
y_pred = [x*coeffs[0]+intercept for x in x_range]

plt.plot(df["odometer"], df["price"], "b.", x_range, y_pred, ":r")
plt.xlabel("odometer")
plt.ylabel("price")
plt.title(eq_str)
plt.show()
