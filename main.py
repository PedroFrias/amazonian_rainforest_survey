# Python libs.:
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import skimage.measure
from scipy import signal
from random import randint
import time
import datetime
from tqdm import tqdm

# Local libs.:
from maphandler import MapHandler

cnn_inputs_dimensions = 32
google_map_dimensions = 512
border_footer_padding = 25
average_size = (cnn_inputs_dimensions) ** 2
model = load_model("model/87_sites_of_deforestation_model.h5")
maphandler = MapHandler()
flag_as_already_processed = np.ones((cnn_inputs_dimensions, cnn_inputs_dimensions))
treshould_for_prediction = 0.15 * cnn_inputs_dimensions ** 2

fig = plt.figure()

def main():
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
    # globlas
    global google_map_dimensions

    # method
    # satellite_image = Image.open('data/maps/test2.jpg')
    satellite_image = maphandler.get_map('satellite', to_rgb=False)
    region_bounds = maphandler.get_bounds()
    satellite_image_as_array = np.array(satellite_image.convert('RGB')) / 25

    # noise reduction
    points_of_interest = satellite_image_as_array[:, :, 0] / satellite_image_as_array[:, :, 1]
    points_of_interest = convolve(points_of_interest, np.ones((5, 5)) / 5 ** 2) > 0.7
    points_of_interest = points_of_interest.astype(int)

    if sum(points_of_interest.flatten()) > 0.1 * google_map_dimensions ** 2:

        sites_of_deforestation = []
        pol_x = np.polyfit([0, google_map_dimensions], [region_bounds[1][0], region_bounds[1][1]], 1)
        pol_y = np.polyfit([google_map_dimensions, 0], [region_bounds[0][0], region_bounds[0][1]], 1)

        # setting iterator up
        indexes_1D = np.arange(google_map_dimensions ** 2).flatten()
        indexes_2D = np.unravel_index(
            indexes_1D,
            (google_map_dimensions,
             google_map_dimensions)
        )

        dataframe = pd.DataFrame()
        dataframe['i'] = indexes_2D[0].astype(int)
        dataframe['j'] = indexes_2D[1].astype(int)
        dataframe['Value on (i, j)'] = points_of_interest.flatten() * np.random.rand(google_map_dimensions ** 2)

        timeout = 22.5
        timeout_start = time.time()

        while time.time() < timeout_start + timeout:
            workbench = [
                np.array(dataframe['Value on (i, j)']).reshape(
                    (google_map_dimensions, google_map_dimensions)
                ),
                np.array(dataframe[dataframe['Value on (i, j)'] > 0.9])
            ]

            index = randint(0, len(workbench[1]))

            try:
                i = np.clip(
                    workbench[1][index][0],
                    a_min=2,
                    a_max=google_map_dimensions - (cnn_inputs_dimensions + 2)
                ).astype(int)

                j = np.clip(
                    workbench[1][index][1],
                    a_min=2,
                    a_max=google_map_dimensions - (cnn_inputs_dimensions + 2)
                ).astype(int)

                plt.imshow(np.clip(workbench[0], a_min=-1, a_max=1), cmap="Blues")
                plt.axis("off")
                plt.pause(1)

                input = satellite_image_as_array[i:i + cnn_inputs_dimensions,
                                                 j:j + cnn_inputs_dimensions]

                filling = sum(points_of_interest[i:i + cnn_inputs_dimensions,
                                                 j:j + cnn_inputs_dimensions].flatten())

                if filling > treshould_for_prediction:
                    prediction = np.round(
                        model.predict(
                            x=np.expand_dims(input, 0),
                            verbose=0
                        )
                    )[0][0]

                    if prediction:
                        # index to coordinates
                        lat, lon = np.polyval(pol_y, i), np.polyval(pol_x, j)
                        sites_of_deforestation.append([
                            lat,
                            lon,
                            filling,
                            datetime.date.today()
                        ])

                        i_range = (
                            np.clip(i - 64,
                                    a_min=0,
                                    a_max=None
                                    ),

                            np.clip(i + 64,
                                    a_min=None,
                                    a_max=google_map_dimensions
                                    )
                            )
                        i_tile_size = abs(i_range[0] - i_range[1])

                        j_range = (
                            np.clip(j - 64,
                                    a_min=0,
                                    a_max=None
                                    ),

                            np.clip(j + 64,
                                    a_min=None,
                                    a_max=google_map_dimensions
                                    )
                            )
                        j_tile_size = abs(j_range[0] - j_range[1])

                        workbench[0][i_range[0]:i_range[1],
                                     j_range[0]:j_range[1]] += create_circular_mask(
                                        i_tile_size,
                                        j_tile_size,
                                        center=None,
                                        radius=None) * (-1)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

                    else:
                        workbench[0][i:i + cnn_inputs_dimensions,
                                     j:j + cnn_inputs_dimensions] += create_circular_mask(
                                        32,
                                        32,
                                        center=None,
                                        radius=None) * (-2)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

            except IndexError:
                pass

            else:
                workbench[0][i:i + cnn_inputs_dimensions,
                             j:j + cnn_inputs_dimensions] += create_circular_mask(
                                32,
                                32,
                                center=None,
                                radius=None) * (-3)

                dataframe['Value on (i, j)'] = workbench[0].flatten()

        plt.close()

        return np.array(sites_of_deforestation).transpose()


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

    coordinates = pd.read_csv("data/unclassified_data.csv")
    coordinates = coordinates.values.tolist()

    sites_of_deforestations = pd.read_csv("data/sites_of_deforestation.csv")

    for coordinate in coordinates:
        lat, lon = coordinate[0], coordinate[1]
        maphandler.set_variables(lat, lon, 15)
        results = main()

        try:
            new_coordinates = pd.DataFrame({
                "Lats": results[0],
                "Lons": results[1],
                "Area": results[2],
                "Date": results[3]
                })

            new_coordinates.reset_index(drop=True)
            sites_of_deforestations = pd.concat([sites_of_deforestations, new_coordinates], axis=0)

        except TypeError:
            pass

        except IndexError:
            pass

    sites_of_deforestations.to_csv("data/sites_of_deforestation.csv")# Python libs.:
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import skimage.measure
from scipy import signal
from random import randint
import time
import datetime
from tqdm import tqdm

# Local libs.:
from maphandler import MapHandler

cnn_inputs_dimensions = 32
google_map_dimensions = 512
border_footer_padding = 25
average_size = (cnn_inputs_dimensions) ** 2
model = load_model("model/87_sites_of_deforestation_model.h5")
maphandler = MapHandler()
flag_as_already_processed = np.ones((cnn_inputs_dimensions, cnn_inputs_dimensions))
treshould_for_prediction = 0.15 * cnn_inputs_dimensions ** 2

fig = plt.figure()

def main():
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
    # globlas
    global google_map_dimensions

    # method
    # satellite_image = Image.open('data/maps/test2.jpg')
    satellite_image = maphandler.get_map('satellite', to_rgb=False)
    region_bounds = maphandler.get_bounds()
    satellite_image_as_array = np.array(satellite_image.convert('RGB')) / 25

    # noise reduction
    points_of_interest = satellite_image_as_array[:, :, 0] / satellite_image_as_array[:, :, 1]
    points_of_interest = convolve(points_of_interest, np.ones((5, 5)) / 5 ** 2) > 0.7
    points_of_interest = points_of_interest.astype(int)

    if sum(points_of_interest.flatten()) > 0.1 * google_map_dimensions ** 2:

        sites_of_deforestation = []
        pol_x = np.polyfit([0, google_map_dimensions], [region_bounds[1][0], region_bounds[1][1]], 1)
        pol_y = np.polyfit([google_map_dimensions, 0], [region_bounds[0][0], region_bounds[0][1]], 1)

        # setting iterator up
        indexes_1D = np.arange(google_map_dimensions ** 2).flatten()
        indexes_2D = np.unravel_index(
            indexes_1D,
            (google_map_dimensions,
             google_map_dimensions)
        )

        dataframe = pd.DataFrame()
        dataframe['i'] = indexes_2D[0].astype(int)
        dataframe['j'] = indexes_2D[1].astype(int)
        dataframe['Value on (i, j)'] = points_of_interest.flatten() * np.random.rand(google_map_dimensions ** 2)

        timeout = 22.5
        timeout_start = time.time()

        while time.time() < timeout_start + timeout:
            workbench = [
                np.array(dataframe['Value on (i, j)']).reshape(
                    (google_map_dimensions, google_map_dimensions)
                ),
                np.array(dataframe[dataframe['Value on (i, j)'] > 0.9])
            ]

            index = randint(0, len(workbench[1]))

            try:
                i = np.clip(
                    workbench[1][index][0],
                    a_min=2,
                    a_max=google_map_dimensions - (cnn_inputs_dimensions + 2)
                ).astype(int)

                j = np.clip(
                    workbench[1][index][1],
                    a_min=2,
                    a_max=google_map_dimensions - (cnn_inputs_dimensions + 2)
                ).astype(int)

                plt.imshow(np.clip(workbench[0], a_min=-1, a_max=1), cmap="Blues")
                plt.axis("off")
                plt.pause(1)

                input = satellite_image_as_array[i:i + cnn_inputs_dimensions,
                                                 j:j + cnn_inputs_dimensions]

                filling = sum(points_of_interest[i:i + cnn_inputs_dimensions,
                                                 j:j + cnn_inputs_dimensions].flatten())

                if filling > treshould_for_prediction:
                    prediction = np.round(
                        model.predict(
                            x=np.expand_dims(input, 0),
                            verbose=0
                        )
                    )[0][0]

                    if prediction:
                        # index to coordinates
                        lat, lon = np.polyval(pol_y, i), np.polyval(pol_x, j)
                        sites_of_deforestation.append([
                            lat,
                            lon,
                            filling,
                            datetime.date.today()
                        ])

                        i_range = (
                            np.clip(i - 64,
                                    a_min=0,
                                    a_max=None
                                    ),

                            np.clip(i + 64,
                                    a_min=None,
                                    a_max=google_map_dimensions
                                    )
                            )
                        i_tile_size = abs(i_range[0] - i_range[1])

                        j_range = (
                            np.clip(j - 64,
                                    a_min=0,
                                    a_max=None
                                    ),

                            np.clip(j + 64,
                                    a_min=None,
                                    a_max=google_map_dimensions
                                    )
                            )
                        j_tile_size = abs(j_range[0] - j_range[1])

                        workbench[0][i_range[0]:i_range[1],
                                     j_range[0]:j_range[1]] += create_circular_mask(
                                        i_tile_size,
                                        j_tile_size,
                                        center=None,
                                        radius=None) * (-1)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

                    else:
                        workbench[0][i:i + cnn_inputs_dimensions,
                                     j:j + cnn_inputs_dimensions] += create_circular_mask(
                                        32,
                                        32,
                                        center=None,
                                        radius=None) * (-2)

                        dataframe['Value on (i, j)'] = workbench[0].flatten()

            except IndexError:
                pass

            else:
                workbench[0][i:i + cnn_inputs_dimensions,
                             j:j + cnn_inputs_dimensions] += create_circular_mask(
                                32,
                                32,
                                center=None,
                                radius=None) * (-3)

                dataframe['Value on (i, j)'] = workbench[0].flatten()

        plt.close()

        return np.array(sites_of_deforestation).transpose()


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

    coordinates = pd.read_csv("data/unclassified_data.csv")
    coordinates = coordinates.values.tolist()

    sites_of_deforestations = pd.read_csv("data/sites_of_deforestation.csv")

    for coordinate in coordinates:
        lat, lon = coordinate[0], coordinate[1]
        maphandler.set_variables(lat, lon, 15)
        results = main()

        try:
            new_coordinates = pd.DataFrame({
                "Lats": results[0],
                "Lons": results[1],
                "Area": results[2],
                "Date": results[3]
                })

            new_coordinates.reset_index(drop=True)
            sites_of_deforestations = pd.concat([sites_of_deforestations, new_coordinates], axis=0)

        except TypeError:
            pass

        except IndexError:
            pass

    sites_of_deforestations.to_csv("data/sites_of_deforestation.csv")

