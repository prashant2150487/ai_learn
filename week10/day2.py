import numpy as np
import matplotlib.pyplot as plt

# define activation functions
def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z=np.exp(z-np.max(z))
    return exp_z/exp_z.sum(axis=0, keepdims = True)

# forward pass function
def forward_pass(X, weight,biases, activation_fucntion):
    z=np.dot(weight,X)+ biases
    a=activation_fucntion(z)
    return a

# exmaple input 
X= np.array(([0.5], [0.8]))
weights = np.array([[0.2,0.4],[0.6,0.1]])
biases = np.array([[0.1],[0.2]])


# perform forword pass with different activation functions
activations ={
    "Sigmoid" : sigmoid,
    "Tanh" : tanh,
    "ReLU": relu,
    "Softmax" : softmax
}

for name , func  in activations.items():
    output = forward_pass(X, weights, biases, func)
    print(f"{name}  Activation output : \n {output}\n")

# define range of input 
z= np.linspace(-10,10,100)

# plot activations function
plt.figure(figsize=(12,18))
plt.plot(z,sigmoid(z), label ="Sigmoid")
plt.plot(z,tanh(z), label="Tanh")
plt.plot(z,relu(z), label= "ReLU")
plt.plot(z, softmax(z), label='Softmax' )
plt.title("Activation function")

plt.xlabel("Input (Z)")
plt.ylabel("Output")
plt.legend()
plt.grid()

plt.show()

    
    



 