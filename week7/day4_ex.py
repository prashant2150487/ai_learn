from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split



# load datatsets
data=load_diabetes() 
df=pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target

# display infromation
print(df.head())
print(df.info())