import pickle
from pathlib import Path
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense, Dropout



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

# normalize the data
X_train= X_train.astype("float32")/255.0
X_test = X_test.astype('float32')/255.0

# one-hot encode the labels 
y_train= to_categorical(y_train,10)
y_test= to_categorical(y_test,10)


print(f"Traning Data shape : {X_train.shape} , label Shapes : {y_train.shape}")
print(f"Test Data shape : {X_test.shape} , label Shapes : {y_test.shape}")


# build the cnn model 
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')

])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)

# Train the model 
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2 
)
# Evaluate the test data set 
test_loss , test_accuracy=  model.evaluate(X_test,y_test)
print(f"Test accuracy  : {test_accuracy:.4f}")

import matplotlib.pyplot as plt


# Plot Accuracy
plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Mode accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title('Mode Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

