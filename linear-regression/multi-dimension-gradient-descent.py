import numpy as np
import matplotlib.pyplot as plt

X = np.array([
[1, 17.9302012052, 94.5205919533],
[1, 97.1446971852, 69.5932819844],
[1, 81.7759007845, 5.73764809688] ])

Y = np.array([ 317, 405, 180])

# Set weights to random values and ensure
# they have a variance of 1/D which is
# optimal but not required for 
# descent.
w = np.random.randn(3) / np.sqrt(3)

learning_rate = 0.000001

# We'll store the mean squared error between Y and Yhat so we can show
# it decreases as we descend the gradient.
errors = []

for t in range(1000):
	YHat = X.dot(w)
	delta = YHat - Y
	gradient = 2 * X.T.dot(delta)
	w = w - (learning_rate * gradient)
	error = delta.dot(delta) / 3
	errors.append(error)

# Plot the mean squared error reducing over the 1000 iterations.
plt.plot(errors)
plt.show()

# Plot the predicted and actual values of Y
plt.plot(YHat, label='prediction')
plt.plot(Y, label='targets')
plt.show()