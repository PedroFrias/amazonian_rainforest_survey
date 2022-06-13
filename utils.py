## Python libs
import numpy as np
from scipy.signal import convolve2d
import skimage.measure


def pooling(inputs, shape, function=np.max):
    return skimage.measure.block_reduce(inputs, shape, function)


def predic(model, inputs):
    # Divide de map in tiles to feed a CNN

    outputs = []
    tile_size = 128
    step = 75

    for i in range(4, inputs.shape[0] - step, step):
        for j in range(4, inputs.shape[1] - step, step):

            tile = np.expand_dims(inputs[i:i + tile_size, j:j + tile_size, :], 0)
            prediction = np.round(model.predict(x=tile, verbose=0))[0][1]
            outputs.append(prediction)

    return np.array(outputs)


def convolve(inputs):
    kernel = kernel = np.array([[-1., -1., -1.], [-1., 8.0, -1.], [-1., -1., -1.]])
    return convolve2d(inputs, kernel, mode='same')




