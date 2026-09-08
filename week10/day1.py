from tensorflow.keras.datasets import mnist , cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data();
print("Training data shape:", X_train_mnist.shape, X_test_mnist.shape)
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data();
print("Training data shape:", X_train_cifar.shape, X_test_cifar.shape)

# Define basic Dense layer
layer = tf.keras.layers.Dense(
    units=10,
    activation="relu"
)

print("Layer created successfully:", layer)

# Visualize MNIST example
plt.imshow(X_train_mnist[0], cmap='gray')
plt.title(f"MNIST Label: {y_train_mnist[0]}")
plt.show()

# visualize CIFAR-10 example
plt.imshow(X_train_cifar[0])
plt.title(f"CIFAR-10 Label: {y_train_cifar[0]}")
plt.show()


