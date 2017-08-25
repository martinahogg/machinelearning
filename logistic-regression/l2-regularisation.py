import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Prepare the sample inputs with three dimensions. The
# first column is a bias of ones. For the first 50
# rows the second and third column values are
# distributed around -2. For the last 50 rows the
# second and third colum values are distributed
# around +2.
X = np.ones((100,3))
X[:,1:] = np.random.randn(100,2)
X[:50,1:] = X[:50,1:] - 2 * np.ones((50,2))
X[50:,1:] = X[50:,1:] + 2 * np.ones((50,2))

# Prepare the sample outputs. For the first 50
# rows the target is 0. For the last 50 rows it
# is one.
T = np.array([0]*50 + [1]*50)

X, T = shuffle(X, T)

# Use the first 50 rows of our samples to train
# the model.
XTrain = X[:-50]
TTrain = T[:-50]

# Training
learning_rate = 0.001
w = np.random.randn(3)
for i in range(1000):
  THatTrain = sigmoid(XTrain.dot(w))
  delta = THatTrain - TTrain
  gradient = 2 * XTrain.T.dot(delta)
  l2 = 10 * 2 * w;
  w = w - (learning_rate * (gradient + l2))

print("Training classification_rate:", np.mean(TTrain == np.round(THatTrain)))

# Testing
XTest = X[-50:]
TTest = T[-50:]
THatTest = sigmoid(XTest.dot(w))
print("Testing classification_rate:", np.mean(TTest == np.round(THatTest)))

plt.scatter(XTest[:,1], XTest[:,2], c=THatTest, s=100, alpha=0.5)
plt.show()