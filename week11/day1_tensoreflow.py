import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

vocab_size = 1000
max_len=200


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words= vocab_size)

X_train = pad_sequences(X_train,maxlen=max_len,padding="post")
X_test = pad_sequences(X_test,maxlen=max_len,padding="post")


print(f"Traning dataset Shape : {X_train.shape}")
print(f"Test dat Shape : {X_train.shape}")

model= Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    SimpleRNN(128, activation = 'tanh', return_sequences = False)
    Dense(1,activation,"sigmoid")
])

model.compile(optimizer ="adam", loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

history= model.fit(X_train,y_train,epoch=5,batch_size=32,validation_split=0.2)


