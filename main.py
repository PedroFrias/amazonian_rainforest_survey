# Python libs.:
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from scipy.signal import convolve2d
import skimage.measure

# Local libs.:
from mapdowloader import MapDownloader

fig, ax = plt.subplots()

cnn_inputs_dimensions = 64
google_map_dimensions = 512
border_footer_padding = 25
min_n_clusters = 2
max_n_clusters = 30


def main(show_results=False):

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

    mapdownloader = MapDownloader()
    lat, lon = -3.2277356201054594, -60.84974901476347
    mapdownloader.set_variables(lat, lon, 15)
    image_gmaps_satellite = mapdownloader.get_map('satellite', to_rgb=False)

    model = load_model("model/4507100.310228641_sites_of_deforestation_model.h5")

    # variable
    # globlas
    global google_map_dimensions

    # method
    # image_gmaps_satellite = Image.open('data/maps/test1.jpg')
    inputs_conv_neuralnetwork = np.array(image_gmaps_satellite.convert('RGB')) / 255
    input_convolution = np.array(image_gmaps_satellite) / 255
    davies_bouldin_scores = list()
    results = list()
    deforestation_sites = 0

    # image processing
    outputs_convolution = convolve(input_convolution, 9)
    point_of_interest = outputs_convolution > 0.7

    dataframe = pd.DataFrame({"Value": point_of_interest.flatten()})
    dataframe['x'] = np.array(dataframe.index % google_map_dimensions)
    dataframe['y'] = np.array(dataframe.index) / google_map_dimensions
    dataframe = dataframe[dataframe['Value'] > 0]
    dataframe.drop(['Value'], axis=1)

    try:
        for n in tqdm(range(min_n_clusters, max_n_clusters)):
            algorithm = (KMeans(n_clusters=n))
            algorithm.fit(dataframe)
            labels = algorithm.labels_
            davies_bouldin_scores.append(davies_bouldin_score(dataframe, labels))

        optimal_n_clustters = davies_bouldin_scores.index(min(davies_bouldin_scores)) + (min_n_clusters + 1)

        kmeans = KMeans(n_clusters=optimal_n_clustters).fit(dataframe)
        centroids = kmeans.cluster_centers_

        # classification
        for centroid in tqdm(centroids):

            # this transformation corrects the inverion caused by the matrice to scatter shifty and make it so is always in bounds.
            i = np.clip(centroid[1],
                        a_min=cnn_inputs_dimensions / 2,
                        a_max=google_map_dimensions - cnn_inputs_dimensions) - 30
            i = i.astype(int)

            j = np.clip(centroid[2],
                        a_min=cnn_inputs_dimensions / 2,
                        a_max=google_map_dimensions - cnn_inputs_dimensions) - 30
            j = j.astype(int)

            input = inputs_conv_neuralnetwork[i:i + cnn_inputs_dimensions, j:j + cnn_inputs_dimensions]
            prediction = model.predict(x=np.expand_dims(input, 0), verbose=0)
            prediction = prediction[0][1] > 0.75  # only 75% plus are considered True.

            if prediction:
                deforestation_sites = deforestation_sites + prediction
                point_of_interest_range = point_of_interest[i:i + cnn_inputs_dimensions, j:j + cnn_inputs_dimensions]
                point_of_interest[i:i + cnn_inputs_dimensions, j:j + cnn_inputs_dimensions] = point_of_interest_range * prediction
                results.append([i, j, prediction])

        # propagation
        if deforestation_sites > 0:
            point_of_interest[
                cnn_inputs_dimensions:google_map_dimensions - cnn_inputs_dimensions,
                cnn_inputs_dimensions:google_map_dimensions - cnn_inputs_dimensions] = False  # avoid interference.

            propagation = pooling(
                point_of_interest * 1,
                int(google_map_dimensions / 3) + 1,
                np.max)

            results.append(propagation)

    except ValueError:
        results = None

    finally:
        if show_results:
            plot_results(inputs_conv_neuralnetwork, results)

        return prediction


def convolve(inputs, size_kernel=9):
    kernel = np.ones((size_kernel, size_kernel)) / size_kernel ** 2
    return convolve2d(inputs, kernel, mode='same')


def pooling(inputs, shape, function=np.mean):
    return skimage.measure.block_reduce(inputs, shape, function)


def plot_results(region_map, results):

    footer = np.zeros((border_footer_padding, google_map_dimensions, 3))
    region_map[google_map_dimensions - border_footer_padding: google_map_dimensions, :] = footer

    if results is not None:
        for n in range(0, len(results) - 1):
            rect = patches.Rectangle((results[n][0], results[n][1]), 64, 64,
                                     linewidth=1,
                                     edgecolor='red' if results[n][2] > 0 else 'lime',
                                     facecolor="none",
                                     alpha=0.75,
                                     label=f'teste')
            ax.add_patch(rect)

        for i, rows in enumerate(range(489, 510, 7)):
            for j, cols in enumerate(range(489, 510, 7)):
                region_map[rows:rows + 7, cols:cols + 7, :] = region_map[rows:rows + 7, cols:cols + 7, :] + results[-1][i][j]

    plt.imshow(region_map)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main(True)

