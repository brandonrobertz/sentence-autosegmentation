from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import * #Dense, Embedding, LSTM, Flatten
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_diagram import ascii


def binary_model(embedding_size=105, window_size=56, window_step=4,
                 lstm_size=5480):
    print('Building model...')
    model = Sequential()
    # 128 character-space (ascii only)
    # best was lstm 2000, embedding 200
    model.add(Embedding(
        128, embedding_size, input_length=window_size
    ))
    model.add(LSTM(
        lstm_size,
        dropout=0.2, recurrent_dropout=0.2
    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam', #Adam(lr=0.001),
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
