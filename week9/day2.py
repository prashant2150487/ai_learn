from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# load datasets
data= load_iris()
X, y = data.data, data.target

# split dataset
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# display dataset info
print(f"Feature Names : {data.feature_names}") 
print(f"Class Names : {data.target_names}")


# Define hyperapremeter
params_grid ={
    'n_estimators': [50, 100, 150],
    'max_depth': [None , 5, 10],
    'min_samples_split': [2,5,10]
}

# initialize Grid search 
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=params_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
# perfom grid_search
grid_search.fit(X_train, y_train)

# evaluate best model
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test , y_pred_grid)
print(f"Best Hyper meter grid search : {grid_search.best_params_}")
print(f"Grid serch accuracy : {accuracy_grid:.4f}")



# define hyper parameter distribution

# Define hyperapremeter
params_dist ={
    'n_estimators': np.arange(50,200,10),
    'max_depth': [None , 5, 10,15],
    'min_samples_split': [2,5,10,20]
}

# initialize Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=params_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Perform random search
random_search.fit(X_train, y_train)
# evaluate best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Best Hypermeter (Random Search) : {random_search.best_params_}")
print(f"Randomized serch accuracy : {accuracy_random:.4f}")