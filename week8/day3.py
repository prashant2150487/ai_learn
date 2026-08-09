from sklearn.datasets  import load_breast_cancer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
 


# load dataset
data = load_breast_cancer();
X,y = data.data, data.target


# split datset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# display datset information
print("Features: ", data.feature_names)
print("Classes: ", data.target_names)


# train grdient boosting model
gb_model =GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# predict

y_pred_gb = gb_model.predict(X_test)


# evaluate performance
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Model Accuracy: {accuracy_gb:.2f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_gb))


# define hypermeter grid
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7]
}

grid_search = GridSearchCV(
    estimator= GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1

)

grid_search.fit(X_train, y_train)

# display best parameters and best score
print(f"Best parmeters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")


# trin random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# predict
# evalaute performance
accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Model Accuracy: {accuracy_rf:.2f}")



