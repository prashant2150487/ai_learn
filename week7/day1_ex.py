import pandas as pd


#load titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


 
# display dataset 
df=pd.read_csv(url)

print("Dataset info: \n")
print(df.info())


# preview first few rows of the dataset
print("First few rows of the dataset:")
# print(df.head())



# separate the feature and target variable
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
# print("Categorical features: \n", categorical_features.tolist())
# print("Numerical features: \n", numerical_features.tolist())


for col in categorical_features:
    print(f"{col}: {df[col].nunique()} unique values")

print(df[numerical_features].describe())    



