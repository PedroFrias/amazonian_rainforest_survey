## Python libs
from keras.optimizers.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os.path
from machine_learning_model import get_model
from random import random
import numpy as np


def main():
    # Upload data

    PATHS = ['data/dataset/0', 'data/dataset/1', 'data/dataset/2']


    train_batches = \
        ImageDataGenerator(rescale=1. / 255).flow_from_directory(
                                        directory=PATHS[0],
                                        target_size=(64, 64),
                                        classes=['0', '1'],
                                        batch_size=10)

    valid_batches = \
        ImageDataGenerator(rescale=1. / 255).flow_from_directory(
                                        directory=PATHS[1],
                                        target_size=(64, 64),
                                        classes=['0', '1'],
                                        batch_size=10)

    test_batches = \
        ImageDataGenerator(rescale=1. / 255).flow_from_directory(
                                        directory=PATHS[2],
                                        target_size=(64, 64),
                                        classes=['0', '1'],
                                        batch_size=10,
                                        shuffle=False)

    images, labels = next(train_batches)

    # sanity check
    plot_images(images, labels)


    # Train model
    model = get_model()
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x=train_batches,
        validation_data=valid_batches,
        epochs=150,
        batch_size=20,
        verbose=2
    )

    # predictions = model.predict(test_batches, verbose=0)
    # predictions = np.round(predictions)

    model.save(f'model/{random()*10000000}_sites_of_deforestation_model.h5')


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
