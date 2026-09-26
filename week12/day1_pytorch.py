import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences  # fixed import & name

# Hyperparameters
vocab_size = 10000       # must match imdb.load_data and model
max_len = 200
embedding_dim = 128
hidden_dim = 128
output_dim = 1
batch_size = 32
lr = 0.001               # 0.01 is usually too large for Adam
epochs = 5

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Convert to PyTorch tensors and create DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model
class RNNModel(nn.Module):  # fixed: nn.Module, not nn.Modules
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # fixed: use self.embedding_dim, not self.embedding_dim undefined
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)           # (batch, seq_len, embedding_dim)
        output, hidden = self.rnn(embed)    # hidden: (1, batch, hidden_dim)
        # hidden.squeeze(0) -> (batch, hidden_dim)
        out = self.fc(hidden.squeeze(0))    # (batch, output_dim)
        return torch.sigmoid(out).squeeze(-1)  # (batch,)

# Instantiate model (vocab_size must match data: 10000)
model = RNNModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):  # fixed: was `range(epoch)`
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)  # already squeezed to (batch,)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

train_rnn(model, train_loader, criterion, optimizer, epochs=epochs)

def evaluate_rnn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.long)
        y_tensor = torch.tensor(y_test, dtype=torch.float32)

        predictions = model(x_tensor)  # (num_samples,)
        loss = criterion(predictions, y_tensor)
        preds_binary = (predictions > 0.5).float()
        accuracy = (preds_binary == y_tensor).float().mean().item()

    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

evaluate_rnn(model, X_test, y_test)