from tarfile import data_filter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X=np.random.rand(100,1)*100
Y= 3*X + np.random.rand(100,1)*2
print([X,Y])
# split data
X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.2 , random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# model prediction
y_pred= model.predict(X_test)
 



print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])



plt.scatter(X, Y, color="blue", label= "Data Points")
plt.plot(X_test, y_pred, color="red", label= "Predict")
plt.xlabel("X",)
plt.ylabel("Y")
plt.legend()
plt.show()


# evalute perfromance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:",mse)
print("R2 score:", r2)