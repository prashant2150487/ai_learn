from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier


#Load dataset
data=load_iris()

# print(data.data)

X,y=data.data, data.target
print(X,y)



# initialize classifire
model=RandomForestClassifier(random_state=42)
print(model)

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model,X,y,cv=kf)
print("Cross-validation scores:", scores)
print("Mean Accuracy:", scores.mean())


