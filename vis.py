import os
import numpy as np
import io
import shapely
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from PIL import Image
from gg_data import *


def show_example(example, predict=None, fname=None):
    image = Image.open(io.BytesIO(example['image']))
    image = np.asarray(image)
    lat = example['latitude']
    long = example['longitude']
    country = get_country(lat, long)

    if predict is None:
        fig, axs = plt.subplots()
        ax = axs
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})
        ax = axs[0]
        pred = predict(image)
        df = pd.DataFrame(pred.items(), columns=['name', 'prob'])
        df = countries.merge(df, how='left', on='name')
        df['prob'] = df['prob'].fillna(0)
        df.plot(column='prob', cmap='viridis', ax=axs[1], legend=True)
        point = gpd.GeoSeries([shapely.Point(long, lat)], crs='EPSG:4326')
        point.plot(ax=axs[1], color='red', markersize=50, label="True Location")
        axs[1].axis('off')

    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{lat}, {long}")
    fig.suptitle(f"{example['id']}: {country}")
    if fname is None:
        fname = example['id'].replace('/','_')
    fig.savefig(f"/users/{uname}/scratch/GeoGoat/vis/{fname}")

if __name__ == '__main__':
    unpacked = open_shard('0')
    show_example(unpacked[0])
    np.random.seed(42)

    def dummy_predict(image):
        countries = ['United States', 'India', 'China', 'Indonesia', 'Pakistan', 'Nigeria', 'Brazil',
            'Bangladesh', 'Russia', 'Mexico', 'Ethiopia', 'Japan']
        dist = np.ones(len(countries))/len(countries)
        dist = np.random.dirichlet(dist) # random distribution
        pred = {}
        for prob, country in zip(dist, countries):
            pred[country] = prob
        return pred
    show_example(unpacked[1], dummy_predict)
    show_example(unpacked[2], dummy_predict)
    show_example(unpacked[3], dummy_predict)

    print("Done with visualization")
