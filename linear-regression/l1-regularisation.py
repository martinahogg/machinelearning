import numpy as np
import matplotlib.pyplot as plt

# Create some training samples

# To demonstrate L1 regularisation we fabricate training data
# with 50 dimensions in our X matrix, where only 3 of which 
# contribute significantly to the values in our Y vector.

# Construct X
X = (np.random.random((50,50)) - 0.5) * 10

# Construct Y
actualW = np.array([1, 0.5, -0.5] + [0]*47)
Y = X.dot(actualW) + np.random.randn(50) * 0.5

# Use gradient descent with L1 regularisation to derive the
# wieghts from the training samples.
w = np.random.randn(50) / np.sqrt(50)
learning_rate = 0.0001
errors = []

for t in range(1000):
  YHat = X.dot(w)
  delta = YHat - Y
  gradient = 2 * X.T.dot(delta)
  l1 = 10 * np.sign(w);
  w = w - (learning_rate * (gradient + l1))
  error = delta.dot(delta) / 50
  errors.append(error)

# Plot the mean squared error reducing over the 1000 iterations.
plt.plot(errors)
plt.show()

# Plot the predicted and actual values of Y
plt.plot(YHat, label='prediction')
plt.plot(Y, label='targets')
plt.show()

# Note how all but the first three derived weights are very
# close to zero.
print(w)
