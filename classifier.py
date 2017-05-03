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
from keras.layers import * #Dense, Embedding, LSTM, Flatten
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_diagram import ascii

import numpy as np
import sys
import cPickle as pickle
import datetime
import time
import os
import random


# window sizes in chars
multiclass = False
# multiclass = True
window_size = 20
window_step = 10
batch_size = 1024
lstm_size = 1500
embedding_size = 250
epochs = 10


def train_test(filename=None, multiclass=False, balance=True):
    print('Loading data...')

    f = open(filename or sys.argv[1], 'r')
    data = f.read()
    total = len(data)
    N = int(len(data)/window_step)

    # balancing classes via downsampling larger class
    larger_class = None
    remove_items = 0
    if balance and not multiclass:
        print("Pre-calculating class balance requirements")
        total_True = 0
        total_False = 0
        data_ix = 0
        for i in range(N):
            chunk = data[data_ix:data_ix+window_size]
            data_ix += window_step
            if "\n" in chunk:
                total_True += 1
            else:
                total_False += 1
        if total_True > total_False:
            larger_class = 'True'
        elif total_True < total_False:
            larger_class = 'False'
        if larger_class is not None:
            remove_items = abs(total_True - total_False)
            N -= remove_items
        print( "T\t", total_True)
        print( "F\t", total_False)
        if larger_class is not None:
            print("R\t", remove_items)
            print("C\t", larger_class)

    # in multiclass we only use windows containing breaks
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

    print("Building zeroed matrices")
    X = np.zeros(shape=(N, window_size), dtype='uint8')
    if not multiclass:
        y = np.zeros(shape=(N, 1), dtype='uint8')
    else:
        y = np.zeros(shape=(N, window_size), dtype='uint8')

    print("Beginning data vectorization")
    space = " " * 5
    data_ix = 0
    i = 0
    while i < N:
        chunk = data[data_ix:data_ix+window_size]

        y_value = "\n" in chunk

        # print(chunk.replace('\n', '\\n'))
        if multiclass and not y_value:
            data_ix += window_step
            continue

        # only balance for binary classification (for now)
        if balance and str(y_value) == larger_class and remove_items > 0:
            # spread the downsampling out across the whole dataset
            if random.randint(0, int(N / 2)) < int(N / 2):
                # TODO: try skipping randomly throughout
                data_ix += window_step
                remove_items -= 1
                if remove_items == 0:
                    print("Classes balanced at", i)
                continue

        if not multiclass:
            y[i] = 1 if y_value else 0
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

    return x_train, y_train, x_test, y_test


def modelname(embedding, lstm, val_acc, multiclass):
    now = time.mktime(datetime.datetime.now().timetuple())
    return '{}_{}_{}_{}_{}.h5'.format(
        'multiclass' if multiclass else 'binary',
        embedding, lstm, val_acc, int(now))


def binary_model():
    print('Building model...')
    model = Sequential()
    # 256 character-space (ascii only)
    # best was lstm 2000, embedding 200
    model.add(Embedding(
        128, 100, input_length=window_size
    ))
    model.add(LSTM(
        2000,
        dropout=0.2, recurrent_dropout=0.2
    ))
    # model.add(Dense(
        # 200,
        # activation='sigmoid',
        # kernel_regularizer='l1_l2',
        # activity_regularizer='l1_l2'
    # ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=0.002),
                metrics=['binary_accuracy'])
    print( '-' * 20, 'Binary Model', '-' * 20)
    print(ascii(model))
    return model


def binary_model_conv_lstm():
    print('Building model...')
    embedding=100
    model = Sequential()

    # character-embeddings
    model.add(Embedding(
        128, embedding, input_length=window_size
    ))

    # reshape into 4D tensor (samples, 1, maxlen, 256)
    #model.add(Reshape((3, embedding, window_size)))
    #model.add(Reshape((3, window_size, embedding)))

    # VGG-like convolution stack
    model.add(Convolution1D(
        filters=32,
        kernel_size=3,
        input_shape=(3, window_size, embedding)
    ))
    model.add(Activation('relu'))
    # model.add(Convolution2D(filters=32, kernel_size=32))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())


    # 256 character-space (ascii only)
    # best was lstm 2000, embedding 200
    model.add(LSTM(
        2000,
        dropout=0.2, recurrent_dropout=0.2
    ))
    # model.add(Dense(
        # 200,
        # activation='sigmoid',
        # kernel_regularizer='l1_l2',
        # activity_regularizer='l1_l2'
    # ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(), #lr=0.01),
                metrics=['binary_accuracy'])
    print( '-' * 20, 'Binary Model', '-' * 20)
    print(ascii(model))
    return model




def multiclass_model():
    print('Building model...')
    model = Sequential()
    # 256 character-space (ascii only)
    model.add(Embedding(
        128, embedding_size, input_length=window_size
    ))
    model.add(LSTM(
        2000, dropout=0.2, recurrent_dropout=0.2
    ))
    model.add(Dense(window_size, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy'])
    print(ascii(model))
    return model


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = train_test(
        multiclass=multiclass, balance=not multiclass)
    print('x_train shape', x_train.shape, 'y_train shape', y_train.shape)
    print('x_train[0]', x_train[0], 'shape', x_train[0].shape)
    print('y_train[0]', y_train[0], 'shape', y_train[0].shape)

    if multiclass:
        model = multiclass_model()
    else:
        # model = binary_model_conv_lstm() # binary_model()
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
        embedding_size, lstm_size, acc, multiclass))

    print('\n', '+' * 20, 'Results', '+' * 20)
    print(ascii(model))
    print('Test score:', score)
    print('Test accuracy:', acc)
