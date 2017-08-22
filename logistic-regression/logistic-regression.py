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

# create train and test sets
X, T = shuffle(X, T)
Xtrain = X[:-50]
Ttrain = T[:-50]
Xtest = X[-50:]
Ttest = T[-50:]

# train loop
learning_rate = 0.001
w = np.random.randn(3)
for i in range(1000):
    Ytrain = sigmoid(Xtrain.dot(w))
    Ytest = sigmoid(Xtest.dot(w))
    w = w - (learning_rate * Xtrain.T.dot(Ytrain - Ttrain))

print("Final train classification_rate:", np.mean(Ttrain == np.round(Ytrain)))
print("Final test classification_rate:", np.mean(Ttest == np.round(Ytest)))

print("Final w: ", w)

plt.scatter(Xtest[:,1], Xtest[:,2], c=Ytest, s=100, alpha=0.5)
plt.show()
