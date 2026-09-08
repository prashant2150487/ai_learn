from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load dataset from the same folder as this script
excel_path = Path(__file__).resolve().parent / "Telco_customer_churn.xlsx"
df = pd.read_excel(excel_path, engine="openpyxl")
# display dataset
print("Detaset info: \n")
print(df.info())
print("\n Class description: \n")
print(df["Churn Label"].value_counts())
print("\n Sample data:\n ", df.head())


# handle missing values
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.fillna({'Total Charges': df['Total Charges'].mean()}, inplace=True)

# encode categorical variables
label_encoder = LabelEncoder()
for column in df.columns:
    if column == 'Churn Label':
        continue
    if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
        df[column] = df[column].fillna('Unknown')
        df[column] = label_encoder.fit_transform(df[column].astype(str))


# encode target variable
df['Churn Label'] = label_encoder.fit_transform(df['Churn Label'])

# scale features
scaler = StandardScaler()
numerical_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# feature and target (drop id/leakage columns that would give away the answer)
X = df.drop(columns=['Churn Label', 'Churn Value', 'Churn Score', 'Churn Reason', 'CustomerID'])
y = df['Churn Label']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train initial model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
#  evaluate initil model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial model accuracy: {accuracy:.4f}")
print("\n Classification report:\n", classification_report(y_test, y_pred))

# define parameter grid
param_dist ={
    'n_estimators': [50, 200, 10],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10 , 20],
    'min_samples_leaf': [1, 2, 4],
}

# initialize Rndomsearch cv
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1
)


# perform random search
random_search.fit(X_train, y_train)

# get best parameters
best_params = random_search.best_params_
print("Best parameters (Randomized searchCV): ", best_params)

# train best model
best_model = random_search.best_estimator_

# predict and evalaute
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned model accuracy: {accuracy_tuned:.4f}")
print("\n Classification report:\n", classification_report(y_test, y_pred_tuned))

# evaluate uisng cross_ validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation accuracy: ", f"{cv_scores.mean():.4f}")




