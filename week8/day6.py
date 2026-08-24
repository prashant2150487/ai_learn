import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# 1. Load Dataset

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# ============================================================
# 2. Dataset Information
# ============================================================

print("Dataset Info:\n")
df.info()

print("\nClass Distribution:\n")
print(df["Class"].value_counts())

print("\nClass Distribution (%):\n")
print(df["Class"].value_counts(normalize=True) * 100)

# ============================================================
# 3. Separate Features and Target
# ============================================================

X = df.drop(columns=["Class"])
y = df["Class"]

# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 5. Check Shapes
# ============================================================

print("\nDataset Shape:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# 6. Check Class Distribution After Split
# ============================================================

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())