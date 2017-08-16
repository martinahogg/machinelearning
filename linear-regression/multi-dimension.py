import numpy as np

# Create some training samples.

# Note the bias 1s added to the start of each raw sample.
# By addign this, we allow the  weight that corresponds 
# to this column in w to play the same role that the
# variable b plays in our single dimension linear 
# regression function.
X = np.array([
[1, 17.9302012052, 94.5205919533],
[1, 97.1446971852, 69.5932819844],
[1, 81.7759007845, 5.73764809688] ])

Y = np.array([ 317, 405, 180])

# Derive the weights that give the lowest error.
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

# Calcualte the predicted y values.
Yhat = np.dot(X, w)

# Calculate a score of the accuracy of our predictions.
d1 = Y - Yhat
d2 = Y - Y.mean()
rsquared = 1 - d1.dot(d1) / d2.dot(d2)
