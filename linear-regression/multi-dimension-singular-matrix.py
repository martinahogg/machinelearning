import numpy as np
import matplotlib.pyplot as plt

# Create some training samples in which np.dot(X.T, X) is a singular matrix
X = np.array([
	[1, 1, 0],
	[1, 1, 0],
	[1, 1, 0],
	[1, 1, 0],
	[1, 1, 0],
	[1, 0, 1],
	[1, 0, 1],
	[1, 0, 1],
	[1, 0, 1],
	[1, 0, 1] ])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Attempt to derive the weights that give the lowest error and 
# observe that this fails because np.dot(X.T, X) is a singular matrix.
# 
# This is one example where we need to use an alternative technique, such
# as gradient descent, to determine the weights that give the lowest
# error.
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
