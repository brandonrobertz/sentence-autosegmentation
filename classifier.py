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
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import sys
import cPickle as pickle
import datetime
import time
import os


# window sizes in chars
multiclass = True
window_size = 20
window_step = 10
batch_size = 512
lstm_size = 4000
embedding_size = 200
epochs = 5


def train_test(multiclass=False):
    print('Loading data...')

    f = open(sys.argv[1], 'r')
    data = f.read()
    total = len(data)

    N = int(len(data)/window_step)
    if multiclass:
        new_N = 0
        data_ix = 0
        for i in range(N):
            chunk = data[data_ix:data_ix+window_size]
            # print(chunk.replace('\n', '\\n'))
            data_ix += window_step
            if multiclass and ("\n" not in chunk):
                continue
            new_N += 1
        print('old N', N, 'new N', new_N)
        N = new_N

    X = np.zeros(shape=(N, window_size), dtype='uint8')
    if not multiclass:
        y = np.zeros(shape=(N, 1), dtype='uint8')
    else:
        y = np.zeros(shape=(N, window_size), dtype='uint8')

    # print(len(data), 'bytes')

    space = " " * 5
    data_ix = 0
    i = 0
    while i < N:
        chunk = data[data_ix:data_ix+window_size]
        # print(chunk.replace('\n', '\\n'))
        if multiclass and "\n" not in chunk:
            data_ix += window_step
            continue

        if not multiclass:
            y[i] = 1 if "\n" in chunk else 0
        else:
            y[i][chunk.index("\n")] = 1

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
        i += 1

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

    if not multiclass:
        print('True', n_true, 'False', n_false, '%', float(n_true) / n_false)
    del data; del X; del y

    return x_train, x_test, y_train, y_test


def modelname(embedding, lstm, val_acc):
    now = time.mktime(datetime.datetime.now().timetuple())
    return '{}_{}_{}_{}.h5'.format(embedding, lstm, val_acc, int(now))


def binary_model():
    print('Building model...')
    model = Sequential()
    # 256 character-space (ascii only)
    # best was lstm 2000, embedding 200
    model.add(Embedding(
        256, embedding_size, input_length=window_size
    ))
    model.add(LSTM(
        lstm_size, dropout=0.1, recurrent_dropout=0.1
    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['binary_accuracy'])
    return model


def multiclass_model():
    print('Building model...')
    model = Sequential()
    # 256 character-space (ascii only)
    model.add(Embedding(
        256, embedding_size, input_length=window_size
    ))
    model.add(LSTM(
        2000, dropout=0.2, recurrent_dropout=0.2
    ))
    model.add(Dense(window_size, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy'])
    return model


x_train, x_test, y_train, y_test = train_test(multiclass=multiclass)
print('x_train shape', x_train.shape, 'y_train shape', y_train.shape)
print('x_train[0]', x_train[0], 'shape', x_train[0].shape)
print('y_train[0]', y_train[0], 'shape', y_train[0].shape)

if multiclass:
    model = multiclass_model()
else:
    model = binary_model()

print('Building model...')
tbCallback = TensorBoard(
    log_dir='./graph',
    write_graph=True,
    write_images=True
)
checkpointCallback = ModelCheckpoint(
    os.path.abspath('.') + '/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    save_best_only = False
)

print('Training ...')
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[tbCallback, checkpointCallback]
)
score, acc = model.evaluate(
    x_test, y_test,
    batch_size=batch_size
)

print('Saving Keras model')
model.save(os.path.abspath('.') + '/models/' + modelname(
    embedding_size, lstm_size, acc))

print('Test score:', score)
print('Test accuracy:', acc)
