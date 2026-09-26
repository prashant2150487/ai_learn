# Long short term Memory (LSTM ) Network
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , SimpleRNN , LSTM, Dense

vocab_size = 10000
max_len=200
(X_train, y_train), (X_test, y_test)= imdb.load_data(num_words = vocab_size)

X_train = pad_sequences(X_train, maxlen=max_len, padding = 'post')
X_test = pad_sequences(X_test, maxlen= max_len, padding = 'post')

print(f"Traning Data Shape : {X_train.shape}")
print(f"Testing Data Shape : {X_test.shape}")
