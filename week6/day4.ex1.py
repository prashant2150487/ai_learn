import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score




# generate synthetic data
np.random.seed(42)
n_samples=100
X=np.random.rand(n_samples,2)*10
y=(X[:,0]*1.5 + X[:,1] >15).astype(int)

print(X)


# create a DataFame
df=pd.DataFrame(X,columns=["Age","Salary"])
df["Purchased"]=y


# split data
X_train , X_test , y_train, y_test  = train_test_split(df[["Age","Salary"]],df["Purchased"],test_size=0.2,random_state=42)


# train the logistics regression model
model=LogisticRegression()
model.fit(X_train,y_train)

# make predictions
y_pred=model.predict(X_test)


# evaluate the performance of the model
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))
print("F1-Score:", f1_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))


# plot decesion boundary
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min , y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


# predict probabailities for grid models
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=z.reshape(xx.shape)



# plot
plt.contourf(xx,yy,Z,alpha=0.8,cmap="coolwarm")
plt.scatter(X[:,0],X[:,1],c=y, edgecolors="k",cmap="coolwarm")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()
