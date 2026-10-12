import sys
import os
import rasterio
import msgpack
import io
import ast
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from shared import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gg_data import *

def continent_dataloaders(full_dataset, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataset

def get_continent_model(num_continents):
    model = resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, num_continents)
    return model

# used GPT to support creation of visualizations in test_continent_model and continent_model_predict
def test_continent_model(model, test_dataset):
    example = test_dataset[0]
    image, label = example
    image = image.unsqueeze(0)

    heatmap, mask = continent_model_predict(model, image, test_dataset.dataset.index_to_continent_map, False)

    masked_pred = np.ma.masked_where(~mask, heatmap)

    cmap = plt.cm.viridis
    cmap_with_gray = cmap.copy()
    cmap_with_gray.set_bad(color='lightgray')

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(masked_pred, cmap=cmap_with_gray)
    cbar = plt.colorbar(im, ax=ax)
    fig.suptitle(str(label))
    fig.savefig(f"/users/{uname}/scratch/GeoGoat/vis/continent.jpg")

# used GPT to support creation of visualizations in test_continent_model and continent_model_predict
def continent_model_predict(model, image, index_to_continent, from_jpg=True, device='cuda'):
    if from_jpg:
        image = flickr_transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    pred = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1).squeeze()
    continent_probs = {index_to_continent[i]: prob.item() for i, prob in enumerate(pred)}
    prob_df = pd.DataFrame(list(continent_probs.items()), columns=["CONTINENT", "prob"])

    continent_probs = dict(zip(prob_df['CONTINENT'], prob_df['prob']))

    latitudes = np.linspace(89.75, -89.75, 360)
    longitudes = np.linspace(-179.75, 179.75, 720)
    mesh_lon, mesh_lat = np.meshgrid(longitudes, latitudes)
    flat_lat = mesh_lat.ravel()
    flat_lon = mesh_lon.ravel()

    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(flat_lon, flat_lat), crs='EPSG:4326')
    joined = gpd.sjoin(points, countries[['CONTINENT', 'geometry']], how='left', predicate='within')

    probs = joined['CONTINENT'].map(continent_probs).fillna(0).values
    heatmap = probs.reshape((360, 720))
    mask = ~joined['CONTINENT'].isna().values.reshape((360, 720))

    return heatmap, mask


if __name__ == '__main__':
    print("Running")

    print("Building index")
    sys.stdout.flush()
    build_flickr_index()

    print("Building dataset")
    sys.stdout.flush()
    dataset = GeoGoatDataset(flickr_transform, get_continent)
    dataset.index_df = dataset.index_df.dropna(subset=['continent'])
    num_continents = len(dataset.continent_to_index_map)
    train_dataloader, val_dataloader, test_dataset = continent_dataloaders(dataset, 64, 2)

    print("Getting model")
    sys.stdout.flush()
    model = get_continent_model(num_continents)

    sys.stdout.flush()
    if os.path.exists('./checkpoints/continent.pt'):
        print("Loading continent model weights from file")
        model.load_state_dict(torch.load('./checkpoints/continent.pt', weights_only=True))
    else:
        print("Training continent model")
        model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, checkpoint='continent.pt')
        print(train_losses)
        print(val_losses)

    test_continent_model(model, test_dataset)

