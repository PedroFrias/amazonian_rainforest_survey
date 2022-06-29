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
from tqdm import tqdm

# Local libs.:
from google_maps import GMaps
from predictions import predictions_for_nearby_areas, remove_redundancy


model = load_model("model/87_sites_of_deforestation_model.h5")
gmaps = GMaps()

# globals
CNN_INPUTS_DIM = 32
GOOGLEMAPS_DIM = 512
PADDING = 25
N = 10
TRESHOULD = 0.05 * (CNN_INPUTS_DIM ** 2)
FIG, (AX1, AX2) = plt.subplots(1, 2)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------------------------------------------------


def main(lat, lon):
    # load data, variables, pre exections, etc.

    # getting info. over the region to be classified
    gmaps.set_variables(lat, lon, 15)
    satellite_image = gmaps.get_map('satellite', to_rgb=False)
    lower_boundary, upper_boundary = gmaps.get_bounds()

    # feature extraction (itarator)
    satellite_image_as_array = np.array(satellite_image.convert('RGB')) / 255
    AX1.imshow(satellite_image_as_array)
    AX1.axis("off")

    points_of_interest = [
        satellite_image_as_array[:, :, 0] > 0.39215,
        satellite_image_as_array[:, :, 0] > satellite_image_as_array[:, :, 1]
    ]
    points_of_interest = points_of_interest[0].astype(int) * points_of_interest[1].astype(int)
    n_points = sum(points_of_interest.flatten())

    # the main execution only takes place if 'points_of_interest' is above a treshould
    if n_points > 0:

        sites_of_deforestation = list()
        group = np.round(lat, 0) * 1000 + np.round(2)
        pol_x = np.polyfit([0, GOOGLEMAPS_DIM], [lower_boundary[1], upper_boundary[1]], 1)
        pol_y = np.polyfit([GOOGLEMAPS_DIM, 0], [lower_boundary[0], upper_boundary[0]], 1)

        # setting iterator up
        indexes_1D = np.arange(GOOGLEMAPS_DIM ** 2).flatten()
        indexes_2D = np.unravel_index(
            indexes_1D,
            (GOOGLEMAPS_DIM, GOOGLEMAPS_DIM)
        )

        dataframe = pd.DataFrame()
        dataframe['i'] = indexes_2D[0].astype(int)
        dataframe['j'] = indexes_2D[1].astype(int)

        # lower the number of points to make it run faster without losing meaning.
        dataframe['Value on (i, j)'] = points_of_interest.flatten() * np.random.rand(GOOGLEMAPS_DIM ** 2)

        # fix a time limit for each iteration
        timeout = 15.5
        timeout_start = time.time()

        # main exectution (classification)
        while time.time() < timeout_start + timeout:

            try:
                site_of_deforestation = 0

                interator = [
                    np.array(dataframe['Value on (i, j)']).reshape(
                        (GOOGLEMAPS_DIM, GOOGLEMAPS_DIM)
                    ),
                    np.array(dataframe[dataframe['Value on (i, j)'] > 0.9])
                ]

                # select a random-ish tile
                index = randint(0, len(interator[1]) - 1)

                i = np.clip(
                    interator[1][index][0],
                    a_min=2,
                    a_max=GOOGLEMAPS_DIM - (CNN_INPUTS_DIM + 2)
                ).astype(int)

                j = np.clip(
                    interator[1][index][1],
                    a_min=2,
                    a_max=GOOGLEMAPS_DIM - (CNN_INPUTS_DIM + 2)
                ).astype(int)

                # tile
                input = satellite_image_as_array[
                    i:i + CNN_INPUTS_DIM,
                    j:j + CNN_INPUTS_DIM
                ]

                # coverage to be tranformed into area
                coverage = sum(
                    points_of_interest[
                        i:i + CNN_INPUTS_DIM,
                        j:j + CNN_INPUTS_DIM
                    ].flatten()
                )

                # the classification is done only if the tile have enougth coverage area
                if coverage > TRESHOULD:

                    # run the inputs trougth a CNN
                    site_of_deforestation = np.round(
                        model.predict(
                            x=np.expand_dims(input, 0),
                            verbose=0
                        ),
                        2
                    )[0][0]

                    if site_of_deforestation > 0.75:

                        # get to i, j index as coordinates
                        lat = np.polyval(pol_y, i)
                        lon = np.polyval(pol_x, j)

                        sites_of_deforestation.append([
                            datetime.date.today(),
                            np.round(lat, 6),
                            np.round(lon, 6),
                            group,
                            site_of_deforestation
                        ])

                # update iterator to mark region to not be processed again

                tile_size = int(CNN_INPUTS_DIM * (1 + site_of_deforestation))

                i_range = (
                    np.clip(i - tile_size, a_min=0, a_max=None),
                    np.clip(i + tile_size, a_min=None, a_max=GOOGLEMAPS_DIM)
                )
                i_tile_size = abs(i_range[0] - i_range[1])

                j_range = (
                    np.clip(j - tile_size, a_min=0, a_max=None),
                    np.clip(j + tile_size, a_min=None, a_max=GOOGLEMAPS_DIM)
                )
                j_tile_size = abs(j_range[0] - j_range[1])

                interator[0][i_range[0]:i_range[1], j_range[0]:j_range[1]] += create_circular_mask(
                    i_tile_size,
                    j_tile_size,
                    center=None,
                    radius=None
                ) * (-1)

                dataframe['Value on (i, j)'] = interator[0].flatten()
                n_points = n_points - coverage

                AX2.imshow(np.clip(interator[0], a_min=-1, a_max=1) * (-1), cmap="Greys")
                AX2.axis("off")
                plt.pause(0.05)

            except ValueError:
                break

        if len(sites_of_deforestation) > 0:

            sites_of_deforestation = np.array(sites_of_deforestation)

            # store classiified data
            dataframe = pd.read_csv("data/dataframe.csv")

            # predictions
            predict = predictions_for_nearby_areas(gmaps, group, sites_of_deforestation)

            # store predictions
            if len(predict) > 0:
                predict = np.array(predict)
                predict = pd.DataFrame({
                "Date": predict[:, 0],
                "Lats": predict[:, 1],
                "Lons": predict[:, 2],
                "Group": predict[:, 3],
                "Validation": predict[:, 4],
                })
                predict.reset_index(drop=True)
                dataframe = pd.concat([dataframe, predict], axis=0)

            sites_of_deforestation = pd.DataFrame({
                "Date": sites_of_deforestation[:, 0],
                "Lats": sites_of_deforestation[:, 1],
                "Lons": sites_of_deforestation[:, 2],
                "Group": sites_of_deforestation[:, 3],
                "Validation": sites_of_deforestation[:, 4],
            })
            sites_of_deforestation.reset_index(drop=True)
            dataframe = pd.concat([dataframe, sites_of_deforestation], axis=0)

            dataframe.to_csv("data/dataframe.csv", index=False)

    else:
        """
        print(f"region centered at ({lat}, {lon}), "
              f"taken {datetime.date.today()} did not meet the criteria.")
        """
        pass

# ----------------------------------------------------------------------------------------------------------------------
# MAIN SUPPORTING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


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

    dataframe = pd.read_csv("data/dataframe.csv")
    dataframe = dataframe[dataframe['Validation'] == 0]
    dataframe = dataframe.values.tolist()
    n_data_points = len(dataframe)

    for i, data in enumerate(dataframe):
        lat, lon = data[1], data[2]
        main(lat, lon)
        remove_redundancy()

