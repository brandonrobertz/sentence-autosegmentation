#!/usr/bin/env python2
'''
Author: Brandon Roberts <brandon@bxroberts.org>
Description:

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
import cPickle as pickle


# window sizes in chars
window_size = 10
window_step = 5
batch_size = 1024 * 3


def train_test():
    print('Loading data...')

    f = open(sys.argv[1], 'r')
    data = f.read()
    total = len(data)

    N = int(len(data)/window_step)
    X = np.zeros(shape=(N, window_size), dtype='uint8')
    y = np.zeros(shape=(N, 1), dtype='uint8')

    print(len(data), 'bytes')

    space = " " * 5
    data_ix = 0
    for i in range(N):
        chunk = data[data_ix:data_ix+window_size]
        # print(chunk.replace('\n', '\\n'))

        y[i] = 1 if "\n" in chunk else 0
        # print('y[', i, '] =', y[i])

        for j in range(window_size):

            if j >= len(chunk):
                break

            c = chunk[j]

            if c == "\n":
                c = " "

            o = ord(c)
            # print('i', i, 'j', j, 'o', o, 'data_ix', data_ix)
            X[i][j] = o

        data_ix += window_step

        if data_ix % 100000 == 0:
            done = (float(data_ix) / total) * 100
            print("Completed: %0.3f%%%s" % (done, space), end='\r')

    f.close()

    print('Splitting test/train')
    x_train = X[:int(len(X) * 0.75)]
    x_test =  X[int(len(X) * 0.75):]
    y_train = y[:int(len(y) * 0.75)]
    y_test =  y[int(len(y) * 0.75):]

    n_true = y.sum()
    n_false = y.shape[0] - n_true
    print('True', n_true, 'False', n_false, '%', float(n_true) / n_false)
    del data; del X; del y

    return x_train, x_test, y_train, y_test


def save_train_test(x_train, x_test, y_train, y_test):
    print('saving data...')
    with open('data.pickle', 'w') as f:
        pickle.dump({
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }, f)


def load_train_test():
    print('loading data...')
    with open('data.pickle', 'r') as f:
        data = pickle.load(f)
    x_train = data["x_train"]
    x_test  = data["x_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = train_test()
# save_train_test(x_train, x_test, y_train, y_test)
#x_train, x_test, y_train, y_test = load_train_test()
print('x_train[0]', x_train[0], x_train[0].shape, 'x_train', x_train.shape)

print('Build model...')
model = Sequential()
# 256 character-space (ascii only)
model.add(Embedding(
    256, 100, input_length=window_size
))
model.add(LSTM(
    1024, dropout=0.2, recurrent_dropout=0.2
))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              # penalize the model for outputting False incorrectly
              # since we have imbalanced data (sentence breaks are
              # much more rare than sentence content)
              #loss_weights={'False': 4.0, 'True': 1.0},
              metrics=['binary_accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

print('Testing on known input (should be line break) [1]')
print(model.predict_proba(np.array([[115,  32,  97, 108, 108,  32, 111, 102, 102, 105]], dtype='uint8')))
print('Should be a non-line break [0]')
print(model.predict_proba(np.array([[116, 104, 101,  32, 102, 114, 111, 122, 101, 110]], dtype='uint8')))
