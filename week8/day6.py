import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# 2. Dataset Information
print("Dataset Info:\n")
df.info()

print("\nClass Distribution:\n")
print(df["Class"].value_counts())

print("\nClass Distribution (%):\n")
print(df["Class"].value_counts(normalize=True) * 100)

# 3. Separate Features and Target
X = df.drop(columns=["Class"])
y = df["Class"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Check Shapes
print("\nDataset Shape:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# 6. Check Class Distribution After Split
print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# =========================================================
# 7. Random Forest WITHOUT SMOTE
# =========================================================

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate
print("\nClassification Report (Without SMOTE):\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test)[:, 1]
)

print(f"ROC-AUC (Without SMOTE): {roc_auc:.4f}")


# =========================================================
# 8. Apply SMOTE ONLY on Training Data
# =========================================================

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)

# Display new class distribution
print("\nClass Distribution After SMOTE:\n")
print(pd.Series(y_resampled).value_counts())


# =========================================================
# 9. Random Forest WITH SMOTE
# =========================================================

rf_model_smote = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

rf_model_smote.fit(X_resampled, y_resampled)


# =========================================================
# 10. Predict Using SMOTE Model
# =========================================================

y_pred_smote = rf_model_smote.predict(X_test)

# Evaluate
print("\nClassification Report (SMOTE):\n")
print(classification_report(y_test, y_pred_smote))

roc_auc_smote = roc_auc_score(
    y_test,
    rf_model_smote.predict_proba(X_test)[:, 1]
)

print(f"ROC-AUC (SMOTE): {roc_auc_smote:.4f}")