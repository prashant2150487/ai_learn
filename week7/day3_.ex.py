import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# One-Hot Encoding
df_one_hot = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)

# Fill missing values
df_one_hot["Age"] = df_one_hot["Age"].fillna(df_one_hot["Age"].median())
df_one_hot["Fare"] = df_one_hot["Fare"].fillna(df_one_hot["Fare"].median())

# Features and target
X = df_one_hot.drop(
    columns=["Survived", "Name", "Ticket", "Cabin", "PassengerId"]
)
y = df_one_hot["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))