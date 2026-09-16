import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# load cifar -10 dataset
(X_train, y_train), (X_test, y_test)= cifar10.load_data()

# normalise pixel value [0,1]
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# one hot encoding target label

y_train  = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print(f"Traning data shape : {X_train.shape}, {y_train.shape}")
print(f"Traning data shape : {X_test.shape}, {y_test.shape}")




