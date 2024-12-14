"""
Some minimal example 
===========================

TTT for Corrupted MNIST

"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the data
plt.plot(x, y, label="Sine Wave")
plt.title("Simple Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
