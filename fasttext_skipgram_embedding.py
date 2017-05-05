# coding: utf-8
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

window_size = 1

# using skipgram embeddings built using fasttext:
# fasttext skipgram -input dataset -output dataset.skipgram
with open('data/dataset.skipgram.vec', 'r') as f:
    data = f.readlines()

word_vectors = {}
samples, dim = data[0].split()

for line in data[1:]:
    word, vec = line.split(' ', 1)
    word_vectors[word] = np.array([
        float(i) for i in vec.split()
    ], dtype='float32')

E = np.zeros(shape=(int(samples), int(dim)), dtype='float32')
word_index = word_vectors.keys()
for ix in range(len(word_index)):
    word = word_index[ix]
    vec = word_vectors[word]
    for j in range(int(dim)):
        E[ix][j] = vec[j]

embedding = Embedding(
    len(word_index),
    int(dim),
    weights=[E],
    input_length=window_size,
    trainable=False
)

model = Sequential()
model.add(embedding)
model.compile('sgd', 'mse', ['accuracy'])

pred = model.predict(np.array([[0]]))
p = pred[0][0]
a = word_vectors[word_index[0]]
print( "Predicted embedding vector", p)
print( "Actual embedding vector", a)
print( "Equal?", p == a)
