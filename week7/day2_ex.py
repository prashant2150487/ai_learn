import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler , StandardScaler



data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Dataset information
print("Dataset info: \n")

print("\nFeature statistics:")
print(X.describe())
print("\nTarget classes:", data.target_names)



# split the dataset
X_train , X_test, y_train ,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# train k-nn classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# pridict and evaluate the model
y_predict = knn.predict(X_test)
print("Accuracy of the model: ", accuracy_score(y_test, y_predict))


# apply min max scaling

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)


# split scaled data
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# train knn classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(x_train_scaled, y_train_scaled)


# predict_and_evaluate
y_predict_scaled = knn_scaled.predict(x_test_scaled)
print("Accuracy of the scaled model: ", accuracy_score(y_test_scaled, y_predict_scaled))








