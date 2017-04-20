#!/usr/bin/env python2
'''
Sentence Segmentation from unstructured, non-punctuated
text. Relies on a dual model system:

    1. For a given window of text, determine the
    probability of a sentence boundary lying inside
    of it.
      a. if no, shift the window forward
      b. if yes, send the window to model 2
    2. For a given text window, determine where the
    sentence boundary lies.

This expands on earlier work:

    Statistical Models for Text Segmentation
    BEEFERMAN, BERGER, LAFFERTY
    School of Computer Science, Carnegie Mellon University
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import numpy as np
import sys


# window sizes in chars
window_size = 10
window_step = 5
batch_size = 32

print('Loading data...')

f = open(sys.argv[1], 'r')
data = f.read()

N = int(len(data)/window_step)
X = np.zeros(shape=(N, window_size), dtype='uint8')
y = np.zeros(shape=(N, 1), dtype='bool_')

print(len(data), 'bytes')

for i in range(0, N, window_step):
    chunk = data[i:i+window_size]
    # print(chunk.replace('\n', '\\n'))

    for j in range(window_size):
        c = chunk[j]
        o = ord(c)
        # print('i', i, 'j', j, 'o', o)
        X[i][j] = o

    y[i] = True if "\n" in chunk else False
    # print('y', y[i])

f.close()

print('Splitting test/train')
x_train = X[:int(len(X) * 0.75)]
x_test =  X[int(len(X) * 0.75):]
y_train = y[:int(len(y) * 0.75)]
y_test =  y[int(len(y) * 0.75):]

del data
del X

print('X', X.shape, 'x_train', x_train.shape, 'y_train', y_train.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(window_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

