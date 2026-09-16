import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path



def load_cifar10_local(data_dir):
    """Load the extracted CIFAR-10 Python batches from disk."""
    data_dir = Path(data_dir)

    def read_batch(file_path):
        with file_path.open("rb") as file:
            batch = pickle.load(file, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.asarray(batch[b"labels"])
        return images, labels

    train_batches = [read_batch(data_dir / f"data_batch_{index}") for index in range(1, 6)]
    X_train = np.concatenate([images for images, _ in train_batches])
    y_train = np.concatenate([labels for _, labels in train_batches])
    X_test, y_test = read_batch(data_dir / "test_batch")
    return (X_train, y_train[:, np.newaxis]), (X_test, y_test[:, np.newaxis])


# Load the extracted CIFAR-10 dataset without downloading it again.
DATA_DIR = Path(__file__).resolve().parent / "data" / "cifar-10-python" / "cifar-10-batches-py"
(X_train, y_train), (X_test, y_test) = load_cifar10_local(DATA_DIR)

# normalise pixel value [0,1]
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# one hot encoding target label

y_train  = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print(f"Traning data shape : {X_train.shape}, {y_train.shape}")
print(f"Traning data shape : {X_test.shape}, {y_test.shape}")

# define the baseline model

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

# compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# display the model summary
model.summary()

# train the baseline model
history = model.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.2)

# evaluate the model on test data
loss,accuracy = model.evaluate(X_test, y_test)
print(f"Base line model test accuracy : {accuracy*100:.2f}%")


# define an improve model
improved_model = Sequential([
    Conv2D(64,(5,5),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(128,(5,5),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])


# compile the improved model with  a learning rate scheduer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# trian the improved model

improved_history = improved_model.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.2,verbose=1)


# Evaluate the improved model on test data
improved_loss, improved_accuracy = improved_model.evaluate(X_test, y_test)
print(f"Improved model test accuracy : {improved_accuracy*100:.2f}%")



# plot the training and validation accuracy for both models
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(improved_history.history['accuracy'], label='Improved Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(improved_history.history['val_accuracy'], label='Improved Validation Accuracy')
plt.title('Training Accuracy over Epochs Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# plot the training and validation loss 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(improved_history.history['loss'], label='Improved Training Loss')
plt.plot(improved_history.history['val_loss'], label='Validating Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()













 