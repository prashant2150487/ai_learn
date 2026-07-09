# Perform EDA and preprocessing 
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# load dataset
data=fetch_california_housing(as_frame=True)
df=data.frame

 

# load telco customer churn dataset

excel_path = Path(__file__).resolve().parent / "Telco_customer_churn.xlsx"
df_telco = pd.read_excel(excel_path)


# Visualize churn distribution
sns.countplot(x="Churn Value", data=df_telco)
plt.title("Churn Distribution")
plt.show()



# handle missing values
df_telco.fillna(df_telco.mean(), inplace=True)



