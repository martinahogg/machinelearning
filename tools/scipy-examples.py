# Gaussian (a.k.a. normal) distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Probability Density Function (PDF)
r = np.random.randn(1000)
p = norm.pdf(r)
plt.scatter(r,p)
plt.title("Probability Density Function (PDF)")
plt.xlabel("r")
plt.ylabel("p")
plt.show()

# Log Probability Density Function (PDF)
r = np.random.randn(1000)
p = norm.logpdf(r)
plt.scatter(r,p)
plt.title("Log Probability Density Function (PDF)")
plt.xlabel("r")
plt.ylabel("p")
plt.show()

# Cumulative Probability Density Function (PDF)
r = np.random.randn(1000)
p = norm.cdf(r)
plt.scatter(r,p)
plt.title("Cumulative Probability Density Function (PDF)")
plt.xlabel("r")
plt.ylabel("p")
plt.show()

# Log Cumulative Probability Density Function (PDF)
r = np.random.randn(1000)
p = norm.logcdf(r)
plt.scatter(r,p)
plt.title("Log Cumulative Probability Density Function (PDF)")
plt.xlabel("r")
plt.ylabel("p")
plt.show()

# Sampling from Gaussian Distribution (1D)
# Standard Deviation=1, Mean=0
r = np.random.randn(1000)
plt.hist(r, bins=100)
plt.title("Sampling from Gaussian Distribution (1D) - Standard Deviation=1, Mean=0")
plt.xlabel("r")
plt.ylabel("frequency")
plt.show()

# Sampling from Gaussian Distribution (1D)
# Standard Deviation=10, Mean=5
r= 10*np.random.randn(1000) + 5
plt.hist(r, bins=100)
plt.title("Sampling from Gaussian Distribution (1D) - Standard Deviation=10, Mean=5")
plt.xlabel("r")
plt.ylabel("frequency")
plt.show()

# Sample from Gaussian Distribution (2D)
# Standard Deviation=1, Mean=0
r = np.random.randn(1000, 2)
plt.scatter(r[:, 0], r[:, 1])
plt.title("Sampling from Gaussian Distribution (2D)")
plt.xlabel("r[:, 0]")
plt.ylabel("r[:. 1]")
plt.show()
