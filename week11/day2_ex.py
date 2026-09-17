# ============================================================
# Part 1: SciPy convolve — edge detection and blur
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

image = np.random.rand(10, 10)

edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)

blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float64) / 9

print("Blur kernel:\n", blur_kernel)
print("\nImage:\n", image)

edge_detected_image = convolve(image, edge_detection_kernel,
                               mode='constant', cval=0.0)
blurred_image = convolve(image, blur_kernel, mode='reflect')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap='gray');               axes[0].set_title("Original image");      axes[0].axis('off')
axes[1].imshow(blurred_image, cmap='gray');       axes[1].set_title("Blurred image");       axes[1].axis('off')
axes[2].imshow(edge_detected_image, cmap='gray'); axes[2].set_title("Edge detected image"); axes[2].axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# Part 2: TensorFlow Conv2D — 3x3, same padding
# ============================================================
import tensorflow as tf

# (batch_size, height, width, channels)
image_tensor = tf.random.normal([1, 10, 10, 1])

conv_layers = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
)
output_tensor = conv_layers(image_tensor)

print(f"Original Shape : {image_tensor.shape}")
print(f"Output Shape   : {output_tensor.shape}")


# ============================================================
# Part 3: PyTorch Conv2d — 3x3, padding=1
# ============================================================
import torch
import torch.nn as nn

# (batch_size, channels, height, width)
image_tensor_pt = torch.randn(1, 1, 10, 10)

conv_layers_pt = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1,
)
output_tensor_pt = conv_layers_pt(image_tensor_pt)

print(f"Original shape: {image_tensor_pt.shape}")
print(f"Output shape:   {output_tensor_pt.shape}")


# ============================================================
# Part 4: TensorFlow Conv2D — larger 5x5 kernel
# ============================================================
conv_layer_large_kernel = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding='same',
)

output_large_kernel = conv_layer_large_kernel(image_tensor)   # ✅ TF tensor
print(f"Large Kernel Output shape : {output_large_kernel.shape}")


# ============================================================
# Part 5: PyTorch Conv2d — stride=2
# ============================================================
conv_layers_stride_2 = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=2,
    padding=1,
)
output_stride_2 = conv_layers_stride_2(image_tensor_pt)

print(f"Stride Output Shape : {output_stride_2.shape}")