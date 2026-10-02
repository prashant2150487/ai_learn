import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset
import numpy as np

# example senence
english_sentence= ["hello", "how are you", "good morning", "good night", "have a nice day", "see you later", "take care", "thank you", "you're welcome", "I'm sorry"]
french_sentence= ["bonjour", "comment ça va", "bon matin", "bonne nuit", "passe une bonne journée", "à plus tard", "prends soin de toi", "merci", "de rien", "je suis désolé"]


def build_vocab(sentences):
    vocab = {"<PAD>":0, "<SOS>":1,"<EOS>":2,"UNK":3}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

english_vocab = build_vocab(english_sentence)
french_vocab = build_vocab(french_sentence)
print(english_vocab)

# Tokenize and pad sentences
def tokenize(sentences , vocab , max_len):
    tokenized =[]
    for sentence in sentences:
        tokens = [vocab.get(word, vocab['<UNK>']) for word in sentences.split()]
        tokens = [vocab["<Pad>"]] + tokens + [vocab["<EOS>"]]
        tokens+= [vocab["<PAD>"]] * (max_len-len(tokens))
        tokenize.append(tokens)

    return np.array(tokens)




