from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

vocab_size = 1000
max_len = 200

# 1. Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size) 

# 2. Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=max_len, padding='post') 
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f"Training Data Shape : {X_train.shape}")
print(f"Test Data Shape : {X_test.shape}")

# 3. Build the Model
model = Sequential([
    # input_length is optional but good practice for summary shapes
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len), 
    # Corrected 'return_sequence' to 'return_sequences'
    SimpleRNN(128, activation='tanh', return_sequences=False), 
    Dense(1, activation='sigmoid')
])

# 4. Compile the Model (Corrected 'optmizer' typo)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train the Model (Corrected 'epoch' to 'epochs')
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 6. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
