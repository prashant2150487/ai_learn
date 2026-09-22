
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

# load cifar-10 dataset from the extracted local batches
data_dir = Path(__file__).resolve().parents[1] / "week10" / "data" / "cifar-10-python" / "cifar-10-batches-py"

(X_train, y_train), (X_test, y_test) = load_cifar10_local(data_dir)

# normalize pixel valued to the rnage [0,1]
X_train= X_train.astype("float32")/255.0
X_test = X_test.astype('float32')/255.0

# one-hot-encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range = 0.1 ,
    height_shift_range=0.1,
    horizontal_flip  = True
)

datagen.fit(X_train)

def create_model():
    model = models.Sequential()
    # convolution layer 1
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32,(3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Dropout(0.25))

    # Convoluction layer 2
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Dropout(0.25))


    # fully connected layer 
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))
    return model


model=create_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using augmented data generator
train_generator = datagen.flow(X_train, y_train, batch_size=64)
history=model.fit(
    train_generator,
    epochs=20,
    validation_data=(X_test,y_test),
    steps_per_epoch=len(train_generator)
)

test_loss,test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy : {test_accuracy:.2f}")

plt.plot(history.history['accuracy'], label='Traning Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Traning and validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Traning Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Traning and validation Loss')
plt.legend()
plt.show()




    



