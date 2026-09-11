import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Reshape and normalize pixel values from [0, 255] to [0.0, 1.0]
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training Data shape : {X_train.shape}")
print(f"Test Data Shape     : {X_test.shape}")

# 4. Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# Display model architecture
model.summary()

# 5. Compile the model (Fixed typo: 'categorical_crossentropy')
model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

# 6. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 7. Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy : {test_accuracy:.4f}")

# 8. Save the model (Recommended format is .keras)
model_path = "mnist_classifier.keras"
model.save(model_path)

# 9. Load the model (Fixed variable name shadowing)
loaded_model = load_model(model_path)

# 10. Verify loaded model performance (Fixed: added .evaluate())
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model Accuracy : {accuracy:.4f}")