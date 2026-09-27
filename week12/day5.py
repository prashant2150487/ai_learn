from pathlib import Path

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np


# ============================================================
# 1. Load IMDB Dataset
# ============================================================

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)


# ============================================================
# 2. Decode Numerical Reviews to Text
# ============================================================

word_index = imdb.get_word_index()

# Reverse the dictionary:
# word -> index  becomes  index -> word
reversed_word_index = {
    value: key
    for key, value in word_index.items()
}

# Decode first 5 reviews
decoded_reviews = [
    " ".join(
        reversed_word_index.get(i - 3, "?")
        for i in review
    )
    for review in X_train[:5]
]


# Print decoded reviews
for i, review in enumerate(decoded_reviews):
    print(f"\nReview {i + 1}:")
    print(review)
    print("Sentiment:", "Positive" if y_train[i] == 1 else "Negative")


# ============================================================
# 3. Pad Sequences
# ============================================================

# Make every review exactly 200 words/tokens long
X_train = pad_sequences(
    X_train,
    maxlen=max_len,
    padding="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=max_len,
    padding="post"
)


# Print dataset shapes
print(f"\nTraining data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")


# ============================================================
# 4. Load GloVe Embeddings
# ============================================================

embedding_index = {}

project_root = Path(__file__).resolve().parents[1]
glove_file = project_root / "data" / "glove.6B" / "glove.6B.100d.txt"

if not glove_file.exists():
    raise FileNotFoundError(f"GloVe file not found: {glove_file}")

# Read the GloVe file
# Each line contains:
# word + 100 numerical values
with open(glove_file, "r", encoding="utf-8") as file:

    for line in file:
        values = line.split()

        word = values[0]

        # Convert embedding values to float32
        coefs = np.asarray(
            values[1:],
            dtype="float32"
        )

        embedding_index[word] = coefs


print(f"Loaded {len(embedding_index)} word vectors from {glove_file}")


# ============================================================
# 5. Prepare Embedding Matrix
# ============================================================

embedding_dim = 100

# Matrix shape:
# vocab_size × embedding_dim
embedding_matrix = np.zeros(
    (vocab_size, embedding_dim)
)

# Match IMDB words with GloVe vectors
for word, i in word_index.items():

    if i < vocab_size:

        embedding_vector = embedding_index.get(word)

        # If GloVe contains this word,
        # put its vector into the matrix
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# ============================================================
# 6. Define LSTM Model with GloVe Embeddings
# ============================================================

lstm_model = Sequential([

    # Convert word IDs into GloVe vectors
    # trainable=False means GloVe weights are frozen
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ),

    # LSTM learns the sequence/context of the review
    LSTM(
        128,
        activation="tanh",
        return_sequences=False
    ),

    # Binary classification:
    # 0 = Negative
    # 1 = Positive
    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 7. Compile Model
# ============================================================

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Display model architecture
lstm_model.summary()


# ============================================================
# 8. Train Model
# ============================================================

history = lstm_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)


# ============================================================
# 9. Evaluate Model
# ============================================================

loss, accuracy = lstm_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"LSTM model with GloVe Test Accuracy: {accuracy:.4f}"
)