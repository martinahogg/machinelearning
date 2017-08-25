import numpy as np

# Inner (or dot) product
a = np.array([1,2])
b = np.array([3,4])
np.inner(a, b)
a.dot(b)

# Outer product
a = np.array([1,2])
b = np.array([3,4])
np.outer(a, b)

# Inverse
m = np.array([[1,2], [3,4]])
np.linalg.inv(m) 

# Inner (or dot) product
m = np.array([[1,2], [3,4]])
minv = np.linalg.inv(m)
m.dot(minv) 

# Diagonal
m = np.array([[1,2], [3,4]])
np.diag(m) 

m = np.array([1,2])
np.diag(m)

# Determinant
m = np.array([[1,2], [3,4]])
np.linalg.det(m) 

# Trace - sum of elements of the diagonal
m = np.array([[1,2], [3,4]])
np.diag(m)
np.diag(m).sum()
np.trace(m)

# Transpose
m = np.array([ [1,2], [3,4] ])
m.T

# Gaussian distribution
m = np.random.randn(2,3)
m 

# Covariance
X = np.random.randn(100,3)
np.cov(X.T)

# Eigen vectors and values
# For symmetric matrix (m == m.T) and hermitian matrix (m = m.H) we use eigh.
m = np.array([
  [ 0.89761228,  0.00538701, -0.03229084],
  [ 0.00538701,  1.04860676, -0.25001666],
  [-0.03229084, -0.25001666,  0.81116126]])

# The first tuple contains three Eigen values. 
# The second tuple contains Eigen vectors stored in columns.
np.linalg.eigh(m)

# Solving linear systems
# The admissions fee at a small far is $1.50 for children an $4.00 for adults. 
# On a certain day 2,200 people enter the fair and $5050 is collected. 
# How many children and how many adults attended.
#
# Let X1 = number of children
# Let X2 = number of adults
# X1 + X2 = 2200
# 1.5X1 + 4X2 = 5050
a = np.array([ [1,1], [1.5,4] ])
b = np.array( [ 2200, 5050] )
np.linalg.solve(a, b)