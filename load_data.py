#!/usr/bin/env python2
from __future__ import print_function
import sys
import numpy as np
import random


def precompute(filename=None, multiclass=False, balance=True):
    """
    Precompute some data that we need in order to lazy batch our data from
    disk directly into our model. Returns (larger_class, remove_items, N)
    """

    f = open(filename or sys.argv[1], 'r')

    # count the total number of windows
    N = 0
    # balancing classes via downsampling larger class
    larger_class = None
    # count how many of larger class to remove to balance
    remove_items = 0

    if balance and not multiclass:
        print("Pre-calculating class balance requirements")
        total_True = 0
        total_False = 0
        text = f.read(window_step)
        while True:
            # chunk = data[data_ix:data_ix+window_size]
            chunk = f.read(window_step)
            if not chunk:
                break

            text += chunk
            text = text[-window_size:]

            if "\n" in chunk:
                total_True += 1
            else:
                total_False += 1

            N += 1

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

    print("N total windows", N)

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

    f.close()

    return larger_class, remove_items, N


def gen_training_data(filename=None, multiclass=False, balance=True,
                      larger_class=None, remove_items=0, N=0):
    # NOTE: this only works with balance set to True
    print('Loading data...')

    f = open(filename or sys.argv[1], 'r')
    print("Beginning data vectorization")
    space = " " * 5

    # window_size = 3
    # window_step = 2
    # batch_size = 4
    #
    # ...
    #   ,..
    #     ,..
    #       ,..

    EOF = False
    while not EOF:
        X = np.zeros(shape=(batch_size, window_size), dtype='uint8')
        if not multiclass:
            y = np.zeros(shape=(batch_size, 1), dtype='uint8')
        else:
            y = np.zeros(shape=(batch_size, window_size), dtype='uint8')

        text = f.read(window_step)
        if not text:
            break

        batch_i = 0
        while batch_i < batch_size:

            piece = f.read(window_step)
            if not piece:
                EOF = True
                break

            # pad the last window if we pull up short
            # this is required because of the splicing below
            if len(piece) < window_step:
                piece += ' ' * (len(piece) - window_step)

            text += piece
            chunk = text[-window_size:]

            y_value = "\n" in chunk

            if multiclass and not y_value:
                continue

            # only balance for binary classification (for now)
            # do balancing randomly so we get a better mix of classes
            if balance and str(y_value) == larger_class and remove_items > 0:
                # spread the downsampling out across the whole dataset
                if (random.random() * N) < (N / 2):
                    remove_items -= 1
                    if remove_items == 0:
                        print("Classes balanced at", i)
                    continue

            if not multiclass:
                y[batch_i] = 1 if y_value else 0
            else:
                y[batch_i][chunk.index("\n")] = 1

            for j in range(window_size):
                if j >= len(chunk):
                    break

                c = chunk[j]

                if c == "\n":
                    c = " "

                o = ord(c)
                # print('i', i, 'j', j, 'o', o, 'data_ix', data_ix)
                X[batch_i][j] = o

            batch_i += 1

        # done = (float(data_ix) / total) * 100
        # print("Completed: %0.3f%%%s" % (done, space), end='\r')

        yield X, y

    f.close()

    # print('Splitting test/train')
    # x_train = X[:int(len(X) * 0.75)]
    # x_test =  X[int(len(X) * 0.75):]
    # y_train = y[:int(len(y) * 0.75)]
    # y_test =  y[int(len(y) * 0.75):]

    # n_true = y.sum()
    # n_false = y.shape[0] - n_true

    # if not multiclass:
    #     print('True', n_true, 'False', n_false, '%', float(n_true) / n_false)
    # del data; del X; del y

    # return x_train, y_train, x_test, y_test

