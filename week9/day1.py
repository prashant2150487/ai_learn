import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics  import accuracy_score, classification_report


# load dataset
data= load_iris()
X, y = data.data , data.target


# spli data set
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=42)

# display dataset info
print("Feature Names:", data.feature_names)
print("Class Names:", data.target_names)


# train randomforest with default hypermeter
rf_default= RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Predict and evaluate

y_predict_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_predict_default)

print(f"Default Model accuracy: {accuracy_default:.4f}" )
print("\n CLassification Report : \n" , classification_report(y_test, y_predict_default))


# train random forest with adjusted hypermeter
rf_tuned = RandomForestClassifier(
    n_estimators=600,
    max_depth=5,
    random_state=42
)
rf_tuned.fit(X_train, y_train)

# pridict and evalaute
y_predict_tuned=rf_tuned.predict(X_test)
accuracy_tuned= accuracy_score(y_test, y_predict_tuned)
print(f"tuned model accuracy : {accuracy_tuned:.4f}")
print("\n Classification report : \n ", classification_report(y_test, y_predict_tuned))