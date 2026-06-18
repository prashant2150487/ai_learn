import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



def sigmoid(x):
    return 1/(1+np.exp(-x))

# generate value
z=np.linspace(-10,10,100)
print(z)
sigmoid_values=sigmoid(z)
print(sigmoid_values)


# plot
plt.plot(z,sigmoid_values)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()
plt.show()