import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# 1. Define transforms
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# 2. Load datasets
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)


# ============================================================
# 3. Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))


# ============================================================
# 4. Define the Neural Network
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Convert 28 x 28 image into 784 values
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        # 28 x 28 -> 784
        x = self.flatten(x)

        # First hidden layer
        x = F.relu(self.fc1(x))

        # Second hidden layer
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x


# Create model
model = NeuralNetwork()

print(model)


# ============================================================
# 5. Define Loss Function and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 6. Training Loop
# ============================================================

def train_model(model, train_loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            # ----------------------------------------
            # 1. Clear previous gradients
            # ----------------------------------------
            optimizer.zero_grad()

            # ----------------------------------------
            # 2. Forward pass
            # ----------------------------------------
            outputs = model(images)

            # ----------------------------------------
            # 3. Calculate loss
            # ----------------------------------------
            loss = criterion(outputs, labels)

            # ----------------------------------------
            # 4. Backpropagation
            # ----------------------------------------
            loss.backward()

            # ----------------------------------------
            # 5. Update weights
            # ----------------------------------------
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )


# Train model
train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=5
)


# ============================================================
# 7. Evaluation Loop
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            # Forward pass
            outputs = model(images)

            # Get predicted class
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")


# Evaluate trained model
evaluate_model(model, test_loader)


# ============================================================
# 8. Save the model
# ============================================================

torch.save(
    model.state_dict(),
    "mnist_model.pth"
)

print("Model saved successfully.")


# ============================================================
# 9. Reload the model
# ============================================================

loaded_model = NeuralNetwork()

loaded_model.load_state_dict(
    torch.load("mnist_model.pth")
)


# ============================================================
# 10. Verify loaded model
# ============================================================

print("Loaded model:")

evaluate_model(
    loaded_model,
    test_loader
)


# ============================================================
# 11. Optional: Continue training with lower learning rate
# ============================================================

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.0001
# )

# train_model(
#     model,
#     train_loader,
#     criterion,
#     optimizer,
#     epochs=5
# )
