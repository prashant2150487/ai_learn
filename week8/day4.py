import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ============================================================
# 1. Load Dataset
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target


# ============================================================
# 2. Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Display Dataset Information
# ============================================================

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of samples  : {X.shape[0]}")
print(f"Number of features : {X.shape[1]}")
print(f"Feature names      : {data.feature_names}")
print(f"Classes            : {data.target_names}")

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape : {X_test.shape}")


# ============================================================
# 4. Convert Dataset into DMatrix
# ============================================================

dtrain = xgb.DMatrix(
    X_train,
    label=y_train
)

dtest = xgb.DMatrix(
    X_test,
    label=y_test
)

print("\n" + "=" * 60)
print("DMATRIX INFORMATION")
print("=" * 60)

print("Training DMatrix:")
print(dtrain)

print("\nTesting DMatrix:")
print(dtest)


# ============================================================
# 5. Train Basic XGBoost Model
# ============================================================

print("\n" + "=" * 60)
print("TRAINING BASIC XGBOOST MODEL")
print("=" * 60)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
    "random_state": 42
}

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100
)


# ============================================================
# 6. Prediction
# ============================================================

y_probability = xgb_model.predict(dtest)

y_pred = (y_probability > 0.5).astype(int)


# ============================================================
# 7. Evaluate Basic XGBoost Model
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("BASIC XGBOOST MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ============================================================
# 8. Define Hyperparameter Grid
# ============================================================

param_grid = {
    "learning_rate": [0.1, 0.01, 0.02],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


# ============================================================
# 9. Initialize XGBClassifier
# ============================================================

xgb_clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# ============================================================
# 10. GridSearchCV
# ============================================================

print("\n" + "=" * 60)
print("RUNNING GRID SEARCH")
print("=" * 60)

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# 11. Train GridSearchCV
# ============================================================

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 12. Display Best Parameters
# ============================================================

print("\n" + "=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)

print("\nBest Parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    f"\nBest Cross-Validation Score: "
    f"{grid_search.best_score_:.2f}"
)


# ============================================================
# 13. Get Best Model
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 14. Predict Using Best Model
# ============================================================

y_pred_best = best_model.predict(
    X_test
)


# ============================================================
# 15. Evaluate Best Model
# ============================================================

best_accuracy = accuracy_score(
    y_test,
    y_pred_best
)

print("\n" + "=" * 60)
print("BEST XGBOOST MODEL PERFORMANCE")
print("=" * 60)

print(f"Test Accuracy: {best_accuracy:.2f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_best,
        target_names=data.target_names
    )
)


# ============================================================
# 16. Compare Models
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Basic XGBoost Accuracy : {accuracy:.2f}")
print(f"GridSearch XGBoost     : {best_accuracy:.2f}")
print(f"Best CV Score          : {grid_search.best_score_:.2f}")