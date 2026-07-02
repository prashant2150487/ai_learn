from xml.parsers.expat import model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# load Iris dataset
data = load_iris()
X,y =data.data, data.target
# split the dataset
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train,X_test)

# train logistic regression model
log_req=LogisticRegression(random_state=42,max_iter=200)
log_req.fit(X_train,y_train)

# predict using logistic regression
y_pred_lr=log_req.predict(X_test)


# evaluate the logistic regression model
accuracy_lr=accuracy_score(y_test,y_pred_lr)
print(f"Accuracy for Logistic Regression: {accuracy_lr:.4f}")

# scale feature
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# expremint with different value of k
for k in range(1,11):
    # initialize k-NN model
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    #  predict the test data
    y_pred=knn.predict(X_test)

    # evaluate the model
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy for k={k}: {accuracy:.4f}")

