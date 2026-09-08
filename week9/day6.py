from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np



# load dataset
data = load_iris()
X, y = data.data , data.target

# split dataset
X_train, X_test , y_train , y_test = train_test_split(X, y , test_size=0.3, random_state=42)
# print("Dataload sucessfully ", data)

# define prameter grdient 
param_grid = {
    'n_estimators': [50,100,150],
    'learning_rate': [0.01,0.1,0.2],
    'max_depth': [3,5,7]

}

# initialize GridSearchCV 
grid_search = GridSearchCV(
    estimator= GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring= 'accuracy',
    cv=5,
    n_jobs=-1
)

# perform grid search
grid_search.fit(X_train, y_train)

best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print("Best Parameters(Grid searchCV): ", best_params_grid)
print("Best cross-validation accuracy (Grid searchCV): ", f"{best_score_grid:.4f}")

# gest best model
best_grid_model = grid_search.best_estimator_

# predict and evaluate
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)
print(f"Test accuracy (Grid searchCV): {accuracy_grid:.4f}")
print("\n Classification report (Grid searchCV):", classification_report(y_test, y_pred_grid))


# define parameter distribution
param_dist = {
    'C': np.logspace(-3, 3, 7),
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    'gamma': ['scale', 'auto'],
    # 'degree': [2, 3, 4]
}

# initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator= SVC(random_state=42),
    param_distributions=param_dist,
    scoring= 'accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42,
    n_iter=30
)

# perform random search 
random_search.fit(X_train, y_train) 

# get best parameters ans score
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print("Best Parameters(Randomized searchCV): ", best_params_random)
print("Best cross-validation accuracy (Randomized searchCV): ", f"{best_score_random:.4f}")

# get best model
best_random_model = random_search.best_estimator_

# predict and evaluate
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f"Test accuracy (Randomized searchCV): {accuracy_random:.4f}")
print("\n Classification report (Randomized searchCV):", classification_report(y_test, y_pred_random)) 
    


