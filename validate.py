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

from keras.models import load_model

import numpy as np
import sys
import os


# window sizes in chars
window_size = 20
window_step = 10
batch_size = 1024
epochs = 3

model_file = sys.argv[1]
text_file = sys.argv[2]

def Xy(text_file):
    print('Loading data...')

    f = open(text_file, 'r')
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

    return X, y


X, y = Xy(text_file)

print('Loading model...')
model = load_model(model_file)

print('Running...')
score, acc = model.evaluate(
    X, y,
    batch_size=batch_size
)

print('Validate score:', score)
print('Validate accuracy:', acc)
