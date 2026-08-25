from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 

# Get the folder where this Python file is located
base_dir = Path(__file__).resolve().parent

# Excel file path
excel_path = base_dir / "Telco_customer_churn.xlsx"
print(excel_path)
df = pd.read_excel(excel_path, engine="openpyxl")
print("Dataset info : \n")
print(df.info())
print("Class distribution : \n")
# print(df['Churn'].value_counts())
print("\n Sample data : \n", df.head())

# handle missing value 
df["Total Charges"]=pd.to_numeric(df["Total Charges"], errors='coerce')
df.fillna({'Total Charges': df['Total Charges'].median()}, inplace=True)
