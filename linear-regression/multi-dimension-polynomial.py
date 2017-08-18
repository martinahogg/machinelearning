import numpy as np
import matplotlib.pyplot as plt
import math

# Create some training samples.

Xraw = np.array([1, 2, 3, 4, 5, 6])

# Our original input sample are 2, 3, 9 & 12. We choose
# to include each value to the power of 0, 1 & 2 in
# separate columns of our input.
#
# By preparing our input values in this way, the algorithm 
# can derive a polynomial (i.e. a curved plane or hyper-plane).
X = np.array([
  [math.pow(Xraw[0], 0), math.pow(Xraw[0], 1), math.pow(Xraw[0],2), math.pow(Xraw[0],3)],
  [math.pow(Xraw[1], 0), math.pow(Xraw[1], 1), math.pow(Xraw[1],2), math.pow(Xraw[1],3)],
  [math.pow(Xraw[2], 0), math.pow(Xraw[2], 1), math.pow(Xraw[2],2), math.pow(Xraw[2],3)],
  [math.pow(Xraw[3], 0), math.pow(Xraw[3], 1), math.pow(Xraw[3],2), math.pow(Xraw[3],3)],
  [math.pow(Xraw[4], 0), math.pow(Xraw[4], 1), math.pow(Xraw[4],2), math.pow(Xraw[4],3)],
  [math.pow(Xraw[5], 0), math.pow(Xraw[5], 1), math.pow(Xraw[5],2), math.pow(Xraw[5],3)] ])

Y = np.array([27, 80, 181, 342, 575, 892])

# Derive the weights that give the lowest error.
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

# Calcualte the predicted y values.
Yhat = np.dot(X, w)

# Calculate a score of the accuracy of our predictions.
d1 = Y - Yhat
d2 = Y - Y.mean()
rsquared = 1 - d1.dot(d1) / d2.dot(d2)

# Plot the training points and predicted line of best fit
plt.scatter(Xraw, Y)
plt.plot(Xraw, Yhat)
plt.title("Multi Dimension Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
