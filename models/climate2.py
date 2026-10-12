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

climate_src = rasterio.open(f"/users/{uname}/scratch/GeoGoat/koppen/1991_2020/koppen_geiger_0p5.tif")
climate_img = climate_src.read(1)

def climate_to_climate2(climate):
    if 1 <= climate <= 3:
        return 1
    if 4 <= climate <= 7:
        return 2
    if 8 <= climate <= 16:
        return 3
    if 17 <= climate <= 28:
        return 4
    if 29 <= climate <= 30:
        return 5
    return 0

def coords_to_climate2(lat, long):
    row, col = climate_src.index(long, lat)
    climate = climate_img[row, col]
    climate2 = climate_to_climate2(climate)
    return climate2

def climate2_dataloaders(full_dataset, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataset

def get_climate2_model():
    model = resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 5+1) # number of climates plus ocean
    return model

def climate2_model_predict(model, image, from_jpg=True, device='cuda'):
    if from_jpg:
        image = flickr_transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    pred = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1).squeeze()
    make_map = np.vectorize(lambda x: pred[climate_to_climate2(x)].item())
    pred_map = make_map(climate_img)
    return pred_map


if __name__ == '__main__':
    print("Running")
    fig, ax = plt.subplots(figsize=(16,8))
    im = ax.imshow(climate_img, cmap='tab20')
    cbar = plt.colorbar(im, ax=ax)
    fig.savefig(f"/users/{uname}/scratch/GeoGoat/vis/climate2.jpg")
    print(coords_to_climate2(41.8, -71.4))
    print(coords_to_climate2(37.8, -71.4))
    sys.stdout.flush()

    print("Getting model")
    sys.stdout.flush()
    model = get_climate2_model()

    print("Building index")
    sys.stdout.flush()
    build_flickr_index()
    print("Building dataset")
    sys.stdout.flush()
    dataset = GeoGoatDataset(flickr_transform, coords_to_climate2)
    train_dataloader, val_dataloader, test_dataset = climate2_dataloaders(dataset, 64, 2)

    sys.stdout.flush()
    if os.path.exists('./checkpoints/climate2.pt'):
        print("Loading climate2 model weights from file")
        model.load_state_dict(torch.load('./checkpoints/climate2.pt', weights_only=True))
    else:
        print("Training climate2 model")
        model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, checkpoint='climate2.pt')
        print(train_losses)
        print(val_losses)

