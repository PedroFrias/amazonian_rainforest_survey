import pandas as pd
import numpy as np
import sklearn.neighbors as neighbors
import matplotlib.pyplot as plt
from google_maps import distance
import datetime
from sklearn.cluster import KMeans


def predictions_for_nearby_areas(gmaps, group, data):

    predictions = list()

    lower_boundary, upper_boundary = gmaps.get_bounds(14)

    surroundings_areas = np.array([
        np.linspace(lower_boundary[0], upper_boundary[0], 3),
        np.linspace(lower_boundary[1], upper_boundary[1], 3)
    ])
    surroundings_areas = np.meshgrid(surroundings_areas[0], surroundings_areas[1])
    surroundings_areas = np.array([
        surroundings_areas[0].flatten(),
        surroundings_areas[1].flatten()
    ]).transpose()

    nbrs = neighbors.NearestNeighbors(
        n_neighbors=1,
        metric=distance
    ).fit(data[:, 1:3])
    distances, indices = nbrs.kneighbors(surroundings_areas)

    for i, dist in enumerate(distances):
        if dist < 1.5:
            predictions.append([
                datetime.date.today(),
                np.round(surroundings_areas[i][0], 6),
                np.round(surroundings_areas[i][1], 6),
                group,
                0
            ])

    return predictions


def remove_redundancy():

    dataframe = pd.read_csv("data/dataframe.csv")
    groups = dataframe["Group"].unique().tolist()
    results = dataframe[dataframe['Validation'] > 0]

    for group in groups:
        group = dataframe[dataframe['Group'] == group]
        predictions = group[group['Validation'] == 0]

        nbrs = neighbors.NearestNeighbors(
            n_neighbors=2,
            metric=distance
        ).fit(
            np.array([
                group['Lats'],
                group['Lons'],
                group.index
            ]).transpose()[:, 0:2]
        )

        distances, indices = nbrs.kneighbors(
            np.array([
                predictions["Lats"],
                predictions["Lons"],
                predictions.index
            ]).transpose()[:, 0:2]
        )

        treshould = 0.65
        indices = distances[:, 1] > treshould

        predictions = predictions.iloc[indices, :]

        if len(predictions) > 0:
            results = pd.concat([results, predictions], axis=0)

    results.to_csv("data/dataframe.csv", index=False)


def clusttering():
    clusters = []

    dataframe = pd.read_csv("data/dataframe.csv")
    dataframe = dataframe[dataframe['Validation'] == 1.0]
    groups = dataframe["Group"].unique().tolist()

    for group in groups:
        group = dataframe[dataframe['Group'] == group]
        x = group["Lons"]
        y = group["Lats"]
        coordinates = np.array([x, y]).transpose()

        kmeans = KMeans(n_clusters=1)
        kmeans.fit(coordinates)
        predict = kmeans.predict(coordinates)
        cluster_center = kmeans.cluster_centers_[0]
        print(cluster_center)

        geometric_center = [
            (max(coordinates[:, 0]) + min(coordinates[:, 0])) / 2,
            (max(coordinates[:, 1]) + min(coordinates[:, 1])) / 2
        ]

        print(geometric_center)


clusttering()