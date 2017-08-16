import numpy as np
import matplotlib.pyplot as plt

# Create some training samples.
X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3, 5, 7, 9, 11, 13])

# Calculate values of a and b that gives least error.
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# Calculate predictes values of Y from the line equation with a and b.
Yhat = a * X + b

# Calculate a score of the accuracy of our predictions.
d1 = Y - Yhat
d2 = Y - Y.mean()
rsquared = 1 - d1.dot(d1) / d2.dot(d2)

# Plot the training points and predicted line of best fit
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.title("Single Dimension Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
