import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

T = np.array([0, 1, 1, 0])

# Complete logistic regression with 2 columns in X.
X = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
])

W = np.random.randn(4)

learning_rate = 0.01
w = np.random.randn(2)
for i in range(1000):
  THat = sigmoid(X.dot(w))
  delta = THat - T
  gradient = 2 * X.T.dot(delta)
  w = w - (learning_rate * gradient)

print("Testing classification rate:", np.mean(T == np.round(THat)))

# Complete logistic regression with the 2 additional columns in X.
multiplied = np.matrix(X[:,0] * X[:,1]).T
ones = np.array([[1]*4]).T
X = np.array(np.concatenate((ones, multiplied, X), axis=1))

W = np.random.randn(4)

learning_rate = 0.01
w = np.random.randn(4)
for i in range(1000):
  THat = sigmoid(X.dot(w))
  delta = THat - T
  gradient = 2 * X.T.dot(delta)
  w = w - (learning_rate * gradient)

print("Testing classification_rate with additional columns:", np.mean(T == np.round(THat)))