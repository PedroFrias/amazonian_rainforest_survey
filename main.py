# Python libs.:
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import skimage.measure
from random import randint
import time
import datetime
from uuid import uuid1 as getID
from PIL import Image

# Local libs.:
from maphandler import MapHandler

model = load_model("model/87_sites_of_deforestation_model.h5")
maphandler = MapHandler()

CNN_INPUTS_DIM = 32
GOOGLEMAPS_DIM = 512
PADDING = 25
TRESHOULD = 0.15 * (CNN_INPUTS_DIM ** 2)
FIG = plt.figure()


def main(lat, lon):
    """
    This function classifies the contents of a RGB image to look for sites of deforestation by first convolving it
    to select areas called points of interest, which basically are regions with high contrats with the surroundings,
    this phase outputs matrice(M) tronsformed into a scatter version for the next step.

    The data is then clusttered, using KMeans with Davies Bouldin - for optimal number of clustters (C). If this method
    return anything the tiles with center C is classified using a CNN.

        - With this aproach only the information that matter is feed trougth to the Neural Network.

    After predictting each tile the propagation is calculate, if needed, by simply reducing (MaxPooling)the sizes of the
    convolution matrice (M) to P with size (3, 3), and P[1, 1] is the region been processed.

    Keywords:
    - deforestation,
    - convolution,
    - clusttering,
    - convolutional neural network,
    - pooling.

    :param show_results:
    :return:
    """

    # variable

    # method
    # atellite_image = Image.open('data/test.jpg')
    satellite_image = maphandler.get_map('satellite', to_rgb=False)
    region_bounds = maphandler.get_bounds()
    satellite_image_as_array = np.array(satellite_image.convert('RGB')) / 25

    # noise reduction
    points_of_interest = satellite_image_as_array[:, :, 0] / satellite_image_as_array[:, :, 1]
    points_of_interest = convolve(points_of_interest, np.ones((5, 5)) / 5 ** 2) > 0.7
    points_of_interest = points_of_interest.astype(int)

    if sum(points_of_interest.flatten()) > 0.1 * GOOGLEMAPS_DIM ** 2:

        sites_of_deforestation = []
        propagation = []
        pol_x = np.polyfit([0, GOOGLEMAPS_DIM], [region_bounds[1][0], region_bounds[1][1]], 1)
        pol_y = np.polyfit([GOOGLEMAPS_DIM, 0], [region_bounds[0][0], region_bounds[0][1]], 1)

        # setting iterator up
        indexes_1D = np.arange(GOOGLEMAPS_DIM ** 2).flatten()
        indexes_2D = np.unravel_index(
            indexes_1D,
            (GOOGLEMAPS_DIM,
             GOOGLEMAPS_DIM)
        )

        dataframe = pd.DataFrame()
        dataframe['i'] = indexes_2D[0].astype(int)
        dataframe['j'] = indexes_2D[1].astype(int)
        dataframe['Value on (i, j)'] = points_of_interest.flatten() * np.random.rand(GOOGLEMAPS_DIM ** 2)

        timeout = 25.5
        timeout_start = time.time()

        while time.time() < timeout_start + timeout:
            workbench = [
                np.array(dataframe['Value on (i, j)']).reshape(
                    (GOOGLEMAPS_DIM, GOOGLEMAPS_DIM)
                ),
                np.array(dataframe[dataframe['Value on (i, j)'] > 0.9])
            ]

            index = randint(0, len(workbench[1]))

            try:
                i = np.clip(
                    workbench[1][index][0],
                    a_min=2,
                    a_max=GOOGLEMAPS_DIM - (CNN_INPUTS_DIM + 2)
                ).astype(int)

                j = np.clip(
                    workbench[1][index][1],
                    a_min=2,
                    a_max=GOOGLEMAPS_DIM - (CNN_INPUTS_DIM + 2)
                ).astype(int)

                input = satellite_image_as_array[
                    i:i + CNN_INPUTS_DIM,
                    j:j + CNN_INPUTS_DIM
                ]

                grid_filling = sum(points_of_interest[
                                   i:i + CNN_INPUTS_DIM,
                                   j:j + CNN_INPUTS_DIM
                ].flatten())

                if grid_filling > TRESHOULD:

                    classification = np.round(
                        model.predict(
                            x=np.expand_dims(input, 0),
                            verbose=0
                        )
                    )[0][0]

                    if classification:
                        # store coordinates
                        coordinates = [np.polyval(pol_y, i), np.polyval(pol_x, j)]
                        sites_of_deforestation.append([
                            datetime.date.today(),
                            coordinates[0],
                            coordinates[1],
                            grid_filling,
                        ])

                        # mark region to not be processed again
                        i_range = (
                            np.clip(i - 64, a_min=0, a_max=None),
                            np.clip(i + 64, a_min=None, a_max=GOOGLEMAPS_DIM)
                        )
                        i_tile_size = abs(i_range[0] - i_range[1])

                        j_range = (
                            np.clip(j - 64, a_min=0, a_max=None),
                            np.clip(j + 64, a_min=None, a_max=GOOGLEMAPS_DIM)
                        )
                        j_tile_size = abs(j_range[0] - j_range[1])

                        workbench[0][
                            i_range[0]:i_range[1],
                            j_range[0]:j_range[1]] += create_circular_mask(
                                i_tile_size,
                                j_tile_size,
                                center=None,
                                radius=None
                        ) * (-1)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

                    else:
                        workbench[0][
                            i:i + CNN_INPUTS_DIM,
                            j:j + CNN_INPUTS_DIM] += create_circular_mask(
                                32,
                                32,
                                center=None,
                                radius=None
                        ) * (-1)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

                    plt.imshow(np.clip(workbench[0], a_min=-1, a_max=1), cmap="Blues")
                    plt.axis("off")
                    plt.pause(1)

            except IndexError:
                pass

        # Nearby areas
        if len(sites_of_deforestation) > 0:

            SITES_OF_DEFORESTATIONS = pd.read_csv("data/sites_of_deforestation.csv")
            PROPAGATION = pd.read_csv("data/propagation.csv")

            sites_of_deforestation = np.array(sites_of_deforestation)
            region_bounds = maphandler.get_bounds()

            neighbors = np.array([
                np.linspace(region_bounds[0][0], region_bounds[0][1], 5)[1:4],
                np.linspace(region_bounds[1][0], region_bounds[1][1], 5)[1:4]
            ])

            neighbors = np.meshgrid(neighbors[0], neighbors[1])
            neighbors = np.array([
                neighbors[0].flatten(),
                neighbors[1].flatten()
            ]).transpose()

            for i, neighbor in enumerate(neighbors):
                if i != 4:
                    treshould = ((lat - neighbor[0]) ** 2 + (lon - neighbor[1]) ** 2) ** (1 / 2)

                    dist = min([
                        sum((site_of_deforestation[1:3] - neighbor) ** 2) ** (1 / 2)
                        for site_of_deforestation in sites_of_deforestation
                    ])

                    if dist < treshould:
                        propagation.append([neighbor[0], neighbor[1]])

            propagation = np.array(propagation)

            try:
                sites_of_deforestation = pd.DataFrame({
                    "ID": getID().hex,
                    "Date": sites_of_deforestation[:, 0],
                    "Lats": sites_of_deforestation[:, 1],
                    "Lons": sites_of_deforestation[:, 2],
                    "Area": sites_of_deforestation[:, 3],
                })

                sites_of_deforestation.reset_index(drop=True)
                sites_of_deforestation = pd.concat([SITES_OF_DEFORESTATIONS, sites_of_deforestation], axis=0)

                propagation = pd.DataFrame({
                    "Lats": propagation[:, 0],
                    "Lons": propagation[:, 1]
                })
                propagation.reset_index(drop=True)
                propagation = pd.concat([PROPAGATION, propagation], axis=0)

                sites_of_deforestation.to_csv("data/sites_of_deforestation.csv", index=False)
                propagation.to_csv("data/propagation.csv", index=False)

            except TypeError:
                pass

        plt.close()


"""
----------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------
"""


def convolve(inputs, kernel):
    return convolve2d(inputs, kernel, mode='same')


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def pooling(inputs, shape, function=np.mean):
    return skimage.measure.block_reduce(inputs, shape, function)


if __name__ == '__main__':

    unclassified_data = pd.read_csv("data/unclassified_data.csv")
    unclassified_data = unclassified_data.values.tolist()

    for coordinate in unclassified_data:
        lat, lon = coordinate[0], coordinate[1]
        maphandler.set_variables(lat, lon, 15)
        main(lat, lon)
