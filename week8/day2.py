from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV




# load datasets
data= load_breast_cancer()
X,y = data.data , data.target



# split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# display dataset infromation
print("Features: ", data.feature_names)
print("Classes", data.target_names)

# Train random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred= rf_model.predict(X_test)


# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# define hyperparameter grid for tuning
param_grid ={
    "n_estimators": [50,100,200],
    "max_depth": [None,10,20],
    "max_features": ["sqrt", "log2", None]
}
grid_search = GridSearchCV(
    estimator= RandomForestClassifier(random_state=42), 
    param_grid=param_grid,
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# disply best parameters and best_score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")