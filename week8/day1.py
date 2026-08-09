from sklearn.datasets import load_iris
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load datasets
data=load_iris()
X,y= data.data, data.target


# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# train individual models
log_model = LogisticRegression()
dt_model =DecisionTreeClassifier()
knn_model = KNeighborsClassifier()


log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)


# creaating voting classifier
ensemble_model = VotingClassifier(
    estimators=[
        
        ('log_reg', log_model),
        ('decision_tree', dt_model),
        ('knn', knn_model)
    ],
    voting='hard'

)

# train essamble  model
ensemble_model.fit(X_train, y_train)
# pridict with assembale
y_pred_essemble = ensemble_model.predict(X_test)


# evaluate accuracy of individual models
accuracy = accuracy_score(y_test, y_pred_essemble)
print(f"Ensemble Model Accuracy: {accuracy:.2f}")

# evaluate individual models
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)



print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Ensemble Model Accuracy: {accuracy:.2f}")









