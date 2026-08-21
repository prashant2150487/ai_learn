import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# load titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df=pd.read_csv(url)

# select features and target
features = ["Pclass", "Sex", "Age", "Fare","Embarked"]
target = "Survived"


# handle missing values
df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)


# ENcode category vriables
label_encoder = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoder[col] = le

# split the dataset into features and target
print(label_encoder)

X= df[features] 
y= df[target]
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape : {X_test.shape}")   

# Train LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# pridict and evaluate the model
lgb_pred=lgb_model.predict(X_test)
print(f" LightGBM Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")



# train catboost model
cat_features = ["Pclass", "Sex", "Embarked"]
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)

# predict and evaluate  the model 
cat_pred = cat_model.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, cat_pred):.4f}")

# Train xgboost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)


# predict and evalute the model
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")


# train catboost without encoding categorial features
cat_model_native = CatBoostClassifier(cat_features=["Sex","Embarked"], verbose=0)
cat_model_native.fit(X_train, y_train)

# pridict and evalute
cat_pred_native = cat_model_native.predict(X_test)
print(f"Catboost Native accuracy: {accuracy_score(y_test, cat_pred_native):.4f} ")






 
