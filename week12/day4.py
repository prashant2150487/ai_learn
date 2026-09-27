# Gated recurrent Unit (GRU)# Long short term Memory (LSTM ) Network
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , SimpleRNN , LSTM, Dense , GRU

vocab_size = 10000
max_len=200
(X_train, y_train), (X_test, y_test)= imdb.load_data(num_words = vocab_size)

X_train = pad_sequences(X_train, maxlen=max_len, padding = 'post')
X_test = pad_sequences(X_test, maxlen= max_len, padding = 'post')

print(f"Traning Data Shape : {X_train.shape}")
print(f"Testing Data Shape : {X_test.shape}")
# print(X_train)

rnn_model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences = False),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer = 'adam',loss ='binary_crossentropy', metrics = ['accuracy'])
rnn_model.summary()
rnn_history = rnn_model.fit(X_train, y_train, epochs= 5 , batch_size=32, validation_split = 0.2)
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)
print(f" RNN test Loss: {rnn_loss:.4f}, Text accuracy: {rnn_accuracy:.4f}")


gru_model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    GRU(128, activation='tanh', return_sequences = False),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer = 'adam',loss ='binary_crossentropy', metrics = ['accuracy'])
gru_model.summary()
gru_history = gru_model.fit(X_train, y_train, epochs= 5 , batch_size=32, validation_split = 0.2)
gru_loss, gru_accuracy = gru_history.evaluate(X_test, y_test)
print(f" LSRM test Loss: {gru_loss:.4f}, Text accuracy: {gru_accuracy:.4f}")





