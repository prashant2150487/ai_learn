import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

df=pd.read_csv(url)

features = df[["total_bill","size"]]
target = df['tip']
# print("features\n",features)
# print("Tip\n",target)



X_train , X_test, y_train, y_test  = train_test_split(features,target,test_size=0.2 , random_state=42)

print("Training Data set", X_train.shape)
print("Testing Data set", X_test.shape)

sns.pairplot(df,x_vars=["total_bill","size"],y_vars="tip",height=5,aspect=0.8,kind="scatter")
plt.title("Feature vs Target relationship")
plt.show()



