

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# pyrefly: ignore [missing-import]
import optuna


# load datasets
data= load_breast_cancer()
X , y = data.data, data.target

# split into traning and test sets
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
print("Scaled Training data : ", X_train.shape)
print("Scaled Test data : ", X_test.shape)


# train a baseline xgboost model  
baselin_model = XGBClassifier(eval_metric="logloss", random_state=42)
baselin_model.fit(X_train, y_train)


# evaluate model  
baseline_pred = baselin_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print(f"Baseline XGboost Accuracy : {baseline_accuracy:.4f}")


# define the objective function

def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators',50,500),
        'learning_rate' : trial.suggest_float('learning_rate',1e-3 ,1e-1 ,log=True),
        'max_depth' : trial.suggest_int('max_depth',3,10),
        'subsample' : trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.6,1.0),
        'gamma' : trial.suggest_float('gamma', 0, 5),
        'reg_alpha' : trial.suggest_float('reg_alpha', 0 , 10),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0 , 10),
    }
    # train xgboost model with suggest params 
    model = XGBClassifier(eval_metric="logloss", random_state=42 ,**params)
    model.fit(X_train,y_train)

    # evaluate model on validation set 
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy
    

# creat optuna study 
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# best parmeterrs 
print("Best Hyperparmeters : ", study.best_params)
print("Best accuracy",study.best_value)


# define parameter grid 
param_grid ={
    "n_estimators" : [50,100,200],
    "learning_rate" : [0.01,0.1,0.2],
    "max_depth" : [3,5,7],
    "subsample" : [0.6,0.8,1.0],

}
# train XGBOOST with grid search 
grid_search =GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train,y_train)
#  best parameter and accuracy 
print("\n\n\n Grid search Best parameter : ", grid_search.best_params_)
print("Grid search Best accuracy : ", grid_search.best_score_)

# define parameter distribution
param_dist ={
    "n_estimators" : [50,100,200,300,400],
    "learning_rate" : [0.01,0.05,0.1, 0.2],
    "max_depth" : [3,5,7,9],
    "subsample" : [0.6,0.8,1.0],
    "colsample_bytree" : [0.6,0.8,0.9,1.0],
}

# train xgboost with random search
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)
random_search.fit(X_train,y_train)
# best parameter and accuracy
print("\n\n\n Random search Best parameter : ", random_search.best_params_)
print("Random search Best accuracy : ", random_search.best_score_)







    

   



