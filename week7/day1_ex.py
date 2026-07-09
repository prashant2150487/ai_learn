import pandas as pd


#load titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


 
# display dataset 
df=pd.read_csv(url)

print("Dataset info: \n")
print(df.info())


# preview first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())


