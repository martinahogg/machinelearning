import numpy as np
import matplotlib.pyplot as plt

# Set the inputs.
# First column is a bias of 1.
# Second column has 5 zeros and five ones.
# Third column has 5 ones and five zeros.
X = np.zeros((10,3))
X[:,0] = 1
X[:5,1] = 1
X[5:,2] = 1

# Set the targets
Y = np.array([0]*5 + [1]*5)

# We cannot use np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# to calcuate our weights because X.T.dot(x) is a
# singular matrix.

# Set weights to random values and ensure
# they have a variance of 1/D which is
# optimal but not required for 
# descent.
w = np.random.randn(3) / np.sqrt(3)

learning_rate = 0.001

# We'll store the mean squared error between Y and Yhat so we can show
# it decreases as we descend the gradient.
costs = []

for t in range(1000):
  YHat = X.dot(w)
  delta = YHat - Y
  gradient = 2 * X.T.dot(delta)
  w = w - learning_rate * gradient
  error = delta.dot(delta)
  errors.append(error)

# Plot the error reducing over the 1000 iterations.
plt.plot(costs)
plt.show()

# Plot the predicted and actual values of Y
plt.plot(YHat, label='prediction')
plt.plot(Y, label='targets')
plt.show()

print(w)

