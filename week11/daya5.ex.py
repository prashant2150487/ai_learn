from turtle import backward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transformations.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the local CIFAR-10 directory relative to this script.
data_root = Path(__file__).resolve().parent.parent / 'week10' / 'data' / 'cifar-10-python'
train_dataset = datasets.CIFAR10(
    root=data_root, train=True, download=False, transform=transform
)
test_dataset = datasets.CIFAR10(
    root=data_root, train=False, download=False, transform=transform
)


# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Traning Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")


# Define  cnn model 
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
print(model)  

# define loss functin and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001 , momentum=0.9)

def train_model(model , train_loader , criterion , optimizer , epochs = 10):
    print("Training the model...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images , labels in train_loader:
            # Zero gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and opitimiztion
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")



train_model(model, train_loader, criterion, optimizer, epochs=10)


# Evaluate the model
def evaluate_model(model, test_loader):
    print("Evaluating the model...")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

evaluate_model(model, test_loader) 



 



