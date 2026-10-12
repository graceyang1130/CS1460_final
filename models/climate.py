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

# climate_src = rasterio.open(f"/users/{uname}/scratch/GeoGoat/koppen/1991_2020/koppen_geiger_1p0.tif")
climate_src = rasterio.open(f"/users/{uname}/scratch/GeoGoat/koppen/1991_2020/koppen_geiger_0p5.tif")
climate_img = climate_src.read(1)

def coords_to_climate(lat, long):
    row, col = climate_src.index(long, lat)
    climate = climate_img[row, col]
    return climate

def climate_dataloaders(full_dataset, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataset

def get_climate_model():
    model = resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 30+1) # number of climates plus ocean
    return model

def test_climate_model(model, test_dataset):
    example = test_dataset[0]
    image, label = example
    image = image.unsqueeze(0)
    pred = climate_model_predict(model, image, False)
    print(pred)
    print(label)
    fig, ax = plt.subplots(figsize=(16,8))
    im = ax.imshow(pred, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    fig.suptitle(str(label))
    fig.savefig(f"/users/{uname}/scratch/GeoGoat/vis/climate.jpg")

def climate_model_predict(model, image, from_jpg=True, device='cuda'):
    if from_jpg:
        image = flickr_transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    pred = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1).squeeze()
    # print(torch.argmax(pred))
    make_map = np.vectorize(lambda x: pred[x].item())
    pred_map = make_map(climate_img)
    return pred_map


if __name__ == '__main__':
    print("Running")
    fig, ax = plt.subplots(figsize=(16,8))
    im = ax.imshow(climate_img, cmap='tab20')
    cbar = plt.colorbar(im, ax=ax)
    fig.savefig(f"/users/{uname}/scratch/GeoGoat/vis/climate.jpg")
    print(coords_to_climate(41.8, -71.4))
    print(coords_to_climate(37.8, -71.4))
    sys.stdout.flush()

    print("Getting model")
    sys.stdout.flush()
    model = get_climate_model()

    print("Building index")
    sys.stdout.flush()
    build_flickr_index()
    print("Building dataset")
    sys.stdout.flush()
    dataset = GeoGoatDataset(flickr_transform, coords_to_climate)
    train_dataloader, val_dataloader, test_dataset = climate_dataloaders(dataset, 64, 2)

    sys.stdout.flush()
    if os.path.exists('./checkpoints/climate.pt'):
        print("Loading climate model weights from file")
        model.load_state_dict(torch.load('./checkpoints/climate.pt', weights_only=True))
    else:
        print("Training climate model")
        model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, checkpoint='climate.pt')
        print(train_losses)
        print(val_losses)

    test_climate_model(model, test_dataset)

