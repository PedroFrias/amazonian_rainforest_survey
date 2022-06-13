## __Python libs__
#  __machine-leanring__
import tensorflow as tf
from tensorflow import keras
from keras.optimizer_v2.adam import Adam
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

#  __plotting__
import matplotlib.pyplot as plt

#  __OS__
import os.path

## __Local libs__
from model import build_model


def main():
    # Upload data

    PATHS = ['data/train', 'data/valid', 'data/test']

    train_batches = \
        ImageDataGenerator(rescale=1. / 255).\
        flow_from_directory(
            directory=PATHS[0],
            target_size=(128, 128),
            classes=['0', '1'],
            batch_size=10
        )

    valid_batches = \
        ImageDataGenerator(rescale=1. / 255).\
        flow_from_directory(
            directory=PATHS[1],
            target_size=(128, 128),
            classes=['0', '1'],
            batch_size=10
        )

    test_batches = \
        ImageDataGenerator(rescale=1. / 255).\
        flow_from_directory(
            directory=PATHS[2],
            target_size=(128, 128),
            classes=['0', '1'],
            batch_size=10,
            shuffle=False
        )

    images, labels = next(train_batches)

    print(images.shape)

    for i, img in enumerate(images):
        print(f'({i})\n{img}\n\n')

    # sanity check
    plot_images(images, labels)

    # Train model


    if os.path.isfile('model/sites_of_deforestation_model.h5') is False:
        model.save('model/sites_of_deforestation_model.h5')


def plot_images(images, labels):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes.flatten()

    for img, ax, lb in zip(images, axes, labels):
        ax.imshow(img)
        ax.axis('off')
        ax.title.set_text(f'{lb}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
