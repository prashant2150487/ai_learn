import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



# generate data 
np.random.seed(42)
X = 2 * np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# visualize data 
plt.scatter(X, y, color='blue')
plt.title("Generate dataset")
plt.xlabel("X")
plt.ylabel("y")
plt.grid()
plt.show()

# initialize parameter
m = 100
theta = np.random.rand(2, 1)
learning_theta = 0.1
iterations = 1000

# add bias term to X 
X_b = np.c_[np.ones((m, 1)), X]

# gradient Descent
for iteration in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_theta * gradients

print("Optimize Parameter (Theta): \n", theta)


# prepare data 
X_tensore= tf.constant(X, dtype=tf.float32)
y_tensore = tf.constant(y, dtype=tf.float32)


# define model 
class LinearModel(tf.Module):
    def __init__(self):
        self.weight=tf.Variable(tf.random.normal([1]))
        self.bias=tf.Variable(tf.random.normal([1]))

    def __call__(self,X):
        return self.weight*X + self.bias

# define loass function
def mse_loss(y_true, y_predict):
    return tf.reduce_mean(tf.square(y_true-y_predict))


# train with SDG
model= LinearModel()
optimizer = tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred =model(X_tensore)
        loss= mse_loss(y_tensore, y_pred)
    gradients=tape.gradient(loss,[model.weight,model.bias])
    optimizer.apply_gradients(zip(gradients, [model.weight, model.bias]))
    if epoch %10==0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

# prepare data 
X_torch= torch.tensor(X,dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)


# define mode 

class LinearModelTorch(nn.Module):
    def __init__(self):
        super(LinearModelTorch,self).__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model_torch = LinearModelTorch()
# define loss fucntion adn optimizer 
criterion = nn.MSELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.1)

# train model 
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_torch(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 ==0:
        print(f"epoch {epoch} , loss: {loss.item():.4f}")


    

