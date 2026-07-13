import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Dataset information
print("Dataset info: \n")

print("\nFeature statistics:")
print(X.describe())
print("\nTarget classes:", data.target_names)
