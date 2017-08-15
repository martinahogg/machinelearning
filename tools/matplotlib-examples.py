import numpy as np
import matplotlib.pyplot as plt

# Create some x & y co-ordiantes to plot
x = np.linspace(0, 10, 11)
y = np.sin(x)

# Points
plt.scatter(x,y)
plt.show()

# Points joined by lines
plt.plot(x, y)
plt.show()

# Create a gaussian distribution to plot
g = np.random.randn(1000)

# Histogram
plt.hist(g)
plt.show()

