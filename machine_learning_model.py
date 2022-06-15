## __Python libs__
#  __machine-leanring__
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D

## __Local libs__


def get_model():

    model = Sequential([
        # Conv. layers.:
        # layer 1.:
        Conv2D(
            filters=32,
            kernel_size=(7, 7),
            activation='relu',
            padding='same',
            input_shape=(64, 64, 3)
        ),

        MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),

        # layer 2.:
        Conv2D(
            filters=32,
            kernel_size=(7, 7),
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),

        # Flattenig.:
        Flatten(),

        # Dense layers
        # layer 1.:
        Dense(64, activation='sigmoid'),
        Dropout(0.2),

        # layer 2.:
        Dense(64, activation='sigmoid'),
        Dropout(0.2),

        # layer 3 (outputs).:
        Dense(2, activation='softmax'),
    ])

    return model
