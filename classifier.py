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

from load_data import precompute, gen_training_data
from models import binary_model, multiclass_model


# window sizes in chars
multiclass = False
# multiclass = True
window_size = 56
window_step = 4
batch_size = 1
lstm_size = 5480
embedding_size = 105
epochs = 1


def modelname(embedding, lstm, val_acc, multiclass):
    now = time.mktime(datetime.datetime.now().timetuple())
    return '{}_{}_{}_{}_{}.h5'.format(
        'multiclass' if multiclass else 'binary',
        embedding, lstm, val_acc, int(now))


if __name__ == "__main__":
    larger_class, remove_items, N = precompute(
        filename=sys.argv[1],
        multiclass=multiclass,
        balance=not multiclass,
        window_size=window_size,
        window_step=window_step
    )
    batch_generator = gen_training_data(
        filename=sys.argv[1],
        multiclass=multiclass,
        balance=not multiclass,
        larger_class=larger_class,
        remove_items=remove_items,
        N=N,
        window_size=window_size,
        window_step=window_step,
        batch_size=batch_size
    )

    # x_test,  y_test  = test( multiclass=multiclass, balance=not multiclass)
    # print('x_train shape', x_train.shape, 'y_train shape', y_train.shape)
    # print('x_train[0]', x_train[0], 'shape', x_train[0].shape)
    # print('y_train[0]', y_train[0], 'shape', y_train[0].shape)

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

    for e in range(epochs):
        # print("x_train", x_train)
        # print("y_train", y_train)
        print('Training ...')
        model.fit_generator(
            batch_generator,
            epochs=1, #epochs,
            steps_per_epoch=N / batch_size,
            # callbacks=[tbCallback, checkpointCallback]
        )

    # validate
    larger_class, remove_items, N = precompute(
        filename=sys.argv[2],
        multiclass=multiclass,
        balance=not multiclass,
        window_size=window_size,
        window_step=window_step
    )
    batch_generator = gen_training_data(
        filename=sys.argv[2],
        multiclass=multiclass,
        balance=False,
        N=N,
        window_size=window_size,
        window_step=window_step,
        batch_size=batch_size
    )
    score, acc = model.evaluate_generator(
        batch_generator,
        steps=N / batch_size
    )

    name = modelname( embedding_size, lstm_size, acc, multiclass)
    print('Saving Keras model', name)
    model.save(os.path.abspath('.') + '/models/' + name)

    print('\n', '+' * 20, 'Results', '+' * 20)
    print(ascii(model))
    print('Test score:', score)
    print('Test accuracy:', acc)
