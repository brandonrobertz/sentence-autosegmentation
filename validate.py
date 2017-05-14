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


if __name__ == "__main__":
    try:
        model_file = sys.argv[1]
        text_file = sys.argv[2]
    except IndexError:
        print('USAGE: ./validate.py [model_file.h5] [text_file.txt]')
        sys.exit(1)
    else:
        print('Using model', model_file, 'and text corpus', text_file)

    larger_class, remove_items, N = precompute(
        filename=text_file,
        multiclass=multiclass,
        balance=False,
        window_step=window_step,
        window_size=window_size
    )

    data_generator = gen_training_data(
        filename=text_file,
        multiclass=multiclass,
        balance=False,
        larger_class=None,
        remove_items=0,
        N=N,
        window_step=window_step,
        window_size=window_size,
        batch_size=batch_size
    )

    print('Loading model...')
    model = load_model(model_file)

    print('Running...')
    score, acc = model.evaluate_generator(
        batch_data,
        num_steps=N/batch_size
    )

    print('Validate score:', score)
    print('Validate accuracy:', acc)
