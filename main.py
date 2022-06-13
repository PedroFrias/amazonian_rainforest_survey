# Python libs.:
import logging, os
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time

# Local libs.:
from map_dowloader import MapDownloader
from utils import convolve, pooling, predic


N_TILES = 16
PADDING = 2
POINTS_OF_INTEREST = None
BOUNDS = None
DEFORESTED_AREAS = []
FIG = plt.figure()


def main(model, mapdownloader):

    lat, lon = -4.363916032425409, -64.12855326762433

    # Large scale survey:
    dataframe, coordinates = pre_evaluation(mapdownloader, lat, lon)
    grid = [np.zeros((N_TILES, N_TILES)), np.zeros((N_TILES, N_TILES))]
    indexes = dataframe.index.tolist()
    regions_for_classification = np.array(dataframe)[:, 0:2]
    n_iterations = int(np.sqrt(N_TILES))

    # Small scale survey:
    for n in tqdm(range(0, n_iterations), position=0, leave=True):
        stop = 0

        for k, region in enumerate(tqdm(regions_for_classification, position=0, leave=True)):
            i, j = np.floor(indexes[k] / N_TILES).astype(int), indexes[k] % N_TILES
            outputs = classify(mapdownloader, model, region)
            grid[0][i][j] += 10

            if sum(outputs) > 0:
                DEFORESTED_AREAS.append(region)

                outputs = outputs.reshape((6, 6))
                outputs = pooling(outputs, 2, np.mean).reshape((3, 3))

                try:
                    grid[1][i - 1:i + 2, j - 1:j + 2] += outputs

                except ValueError:
                    stop += 1

                finally:
                    if stop == len(dataframe):
                        break
            time.sleep(0.01)
        time.sleep(0.01)

        propagation = grid[1] - grid[0]
        dataframe = to_dataframe(coordinates, propagation, 0.75, tail=True)
        indexes = dataframe.index.tolist()
        regions_for_classification = np.array(dataframe)[:, 0:2]


    plot_results()


def pre_evaluation(mapdownloader, lat, lon):

    global POINTS_OF_INTEREST, BOUNDS

    # download map
    mapdownloader.set_variables(lat, lon, 11)
    inputs = mapdownloader.get_map('satellite', save=True, to_rgb=False)
    bounds = mapdownloader.get_bounds()

    outputs = convolve(inputs)

    POINTS_OF_INTEREST = outputs
    BOUNDS = bounds

    outputs = pooling(outputs, 32)
    outputs = outputs.reshape((outputs.shape[0], outputs.shape[1]))
    outputs = outputs[PADDING:len(outputs) - PADDING, PADDING:len(outputs) - PADDING]
    outputs = np.pad(outputs, PADDING, mode='constant')
    outputs = outputs / np.max(outputs)
    outputs = outputs * np.random.randint(2, size=outputs.shape)

    lats = np.linspace(bounds[0][1], bounds[0][0], N_TILES + 2)[1:N_TILES + 1]
    lons = np.linspace(bounds[1][0], bounds[1][1], N_TILES + 2)[1:N_TILES + 1]
    coordinates = np.meshgrid(lons, lats)

    dataframe = to_dataframe(coordinates, outputs, 0.60, tail=True)

    return dataframe, coordinates


def classify(mapdownloader, model, coordinates):
    lat, lon = coordinates[0], coordinates[1]
    mapdownloader.set_variables(lat, lon, 13)
    inputs = mapdownloader.get_map('satellite', save=False)
    outputs = predic(model, inputs)

    plt.imshow(inputs)
    plt.title('true' if sum(outputs) > 0 else 'false')
    plt.draw()
    plt.pause(.0035)
    FIG.clear()

    return outputs


def to_dataframe(coordinates, region_evaluation, thresold, tail=True):
    dataframe = pd.DataFrame({
        "Lats": coordinates[1].flatten(),
        "Lons": coordinates[0].flatten(),
        "Region evaluation": region_evaluation.flatten(),
    })

    dataframe = dataframe[dataframe['Region evaluation'] > thresold]
    if tail:
        dataframe = dataframe.tail(15)

    return dataframe


def plot_results():

    deforested_areas = np.array(DEFORESTED_AREAS)

    plt.plot(deforested_areas[:, 1], deforested_areas[:, 0], '.', color='cornflowerblue')
    plt.show()


if __name__ == '__main__':
    model = load_model('model/sites_of_deforestation_model.h5')
    mapdownloader = MapDownloader()
    main(model, mapdownloader)

