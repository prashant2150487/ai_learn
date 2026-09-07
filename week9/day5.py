import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split , cross_val_score , KFold



# load dataset
url= "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df=pd.read_csv(url)


# display dataset info
print(f"Dataset Info : {df.info}")
print("\n Class Distribution : \n", df['Class'].value_counts())

# define and feture
X= df.drop(columns=['Class'])
y=df['Class']

# split dataset
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# initialize K-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train and evaluate model
rf_model = RandomForestClassifier(random_state=42)
score_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')
print(f"Cross Validation Accuracy : {score_kfold:.4f}")
print(f"Mean Accuracy (K-fold): {score_kfold.mean():.2f}")

# initialize Stratified K-fold
skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# train and evaluate
score_straitfied = cross_val_score(rf_model,X_train,y_train,cv=skf,scoring='accuracy')
print(f"stratified K-fold cross validation scores :  , {score_straitfied.mean():.2f}")
print(f"Mean accuracy (strtified  K-fold) : {score_kfold.mean():.2f}")








