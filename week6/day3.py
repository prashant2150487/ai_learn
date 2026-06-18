import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import PolynomialFeatures
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# genrate synthetic data
np.random.seed(0)
X=np.random.rand(100,1)*10
y=3*X**2+2*X+1+ np.random.randn(100,1)*10
# print(x)
# print(y)


# tranform feature to polynomial 
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly_features.fit_transform(X)
# print(X_poly)

# fir ploynomial regression model
model= LinearRegression()
model.fit(X_poly,y)
y_pred=model.predict(X_poly)




# plot result
plt.scatter(X,y,color="blue", label="Actual data")
plt.scatter(X, y_pred, color="red", label="Predicted data")
plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# evaluate model
mse=mean_squared_error(y,y_pred)

print(f"Mean Squared Error: {mse}")