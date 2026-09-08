from tensorflow.keras.datasets import mnist, cifar10

# load mnist
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print("MNIST dataset loaded successfully", "Training samples:", X_train_mnist.shape, "Testing samples:", X_test_mnist.shape)

# load cifar10
(X_train_cifar10, y_train_cifar10), (X_test_cifar10, y_test_cifar10) = cifar10.load_data()
print("CIFAR-10 dataset loaded successfully", "Training samples:", X_train_cifar10.shape, "Testing samples:", X_test_cifar10.shape)