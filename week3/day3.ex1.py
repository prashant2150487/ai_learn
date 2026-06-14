import select

import pandas as pd

df=pd.read_csv("/home/pranshant/Desktop/project/udemyCourse/AI_learn/week3/iris.csv")
# first five rows
print(df.head())   

print(df.tail())
print(df.info())
select_columns=df[["sepal_length","petal_width"]]
print(select_columns)