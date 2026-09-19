import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, uniform_filter
import tensorflow as tf
import torch
import torch.nn as nn


# ============================================================
# 1. Create sample feature map
# ============================================================

feature_map = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])


# ============================================================
# 2. SciPy filtering
# ============================================================

max_filtered = maximum_filter(
    feature_map,
    size=2,
    mode="constant"
)

average_filtered = uniform_filter(
    feature_map,
    size=2,
    mode="constant"
)


# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(feature_map, cmap="viridis")
axes[0].set_title("Original Feature Map")

axes[1].imshow(max_filtered, cmap="viridis")
axes[1].set_title("Maximum Filter")

axes[2].imshow(average_filtered, cmap="viridis")
axes[2].set_title("Average Filter")

plt.show()


# ============================================================
# 3. TensorFlow Pooling
# ============================================================

# TensorFlow format:
# (batch, height, width, channels)
#
# (1, 4, 4, 1)

input_tensor = tf.constant(
    feature_map.reshape(1, 4, 4, 1),
    dtype=tf.float32
)


# Max pooling
max_pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2,
    padding="valid"
)

max_pooled_tensor = max_pool(input_tensor)


# Average pooling
avg_pool = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=2,
    padding="valid"
)

avg_pooled_tensor = avg_pool(input_tensor)


print("TensorFlow Max Pooling:")
print(tf.squeeze(max_pooled_tensor).numpy())

print("\nTensorFlow Average Pooling:")
print(tf.squeeze(avg_pooled_tensor).numpy())


# ============================================================
# 4. PyTorch Pooling
# ============================================================

# PyTorch format:
# (batch, channels, height, width)
#
# (1, 1, 4, 4)

input_tensor_torch = torch.tensor(
    feature_map,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)


# Max pooling
max_pool_torch = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

max_pooled_torch = max_pool_torch(input_tensor_torch)


# Average pooling
avg_pool_torch = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

avg_pooled_torch = avg_pool_torch(input_tensor_torch)


print("\nPyTorch Max Pooling:")
print(max_pooled_torch.squeeze().numpy())

print("\nPyTorch Average Pooling:")
print(avg_pooled_torch.squeeze().numpy())


# ============================================================
# 5. TensorFlow CNN Example
# ============================================================

model_tf = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2)
    )
])

print("\nTensorFlow Model:")
model_tf.summary()


# ============================================================
# 6. PyTorch CNN Example
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3
        )

        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.pool1(x)

        x = torch.relu(self.conv2(x))

        x = self.pool2(x)

        return x


model_torch = SimpleCNN()

print("\nPyTorch Model:")
print(model_torch)


print("\nOriginal Feature Map:")
print(feature_map)