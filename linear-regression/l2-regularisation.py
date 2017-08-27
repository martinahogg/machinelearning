import numpy as np
import matplotlib.pyplot as plt

# Create some training samples

# To demonstrate L2 regularisations ability to minimize the impact 
# of outliers, we fabricate training data where X and Y are roughly
# linearly related and then insert a few outliers into Y.

# Construct X
X = np.linspace(0, 10, 50)

# Construct Y
Y = 0.5*X + np.random.randn(50)
Y[-1] += 50
Y[-2] += 50 

# Add a bias column to X.
X = np.vstack([np.ones(50), X]).T

# Calculate the weights that give mimumum error using L2 regularisation
# and the corresponding predicated Y values. Use a L2 hyper-parameter 
# value of 1000.
l2 = 1000
w_with_l2 = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
YHat_with_l2 = X.dot(w_with_l2)

# Plot the training data and solutions.
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], YHat_with_l2, label="L2")
plt.legend()
plt.show()