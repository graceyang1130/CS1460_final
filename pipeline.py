import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
from gg_data import *
import sys
sys.path.append(os.path.abspath('models'))
from climate import get_climate_model, climate_model_predict, climate_src
from climate2 import get_climate2_model, climate2_model_predict
from country import get_country_model, country_model_predict
from continent import get_continent_model, continent_model_predict

class GeoGoatEnsemble(torch.nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, heatmaps):
        weights = F.softmax(self.weights, dim=0)
        return torch.sum(weights[:,None,None] * heatmaps, dim=0)

def random_heatmap():
    profile = climate_src.profile.copy()
    height, width = climate_src.height, climate_src.width
    heatmap = np.random.rand(height, width)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    return heatmap

def visualize(image, coords, fname='vis_pipeline.jpg'):
    from_tensor = False
    if type(image) == str:
        image = Image.open(image)
        image = np.asarray(image)
    else:
        from_tensor = True
        tensor = image
        resnet_mean=np.array([0.485, 0.456, 0.406])
        resnet_std=np.array([0.229, 0.224, 0.225])
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = image*resnet_std + resnet_mean

    if coords is not None:
        true_lat, true_lon = coords
        print(true_lat, true_lon)
    if not from_tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    preds = get_predictions(tensor, from_jpg=False)
    heatmap, (pred_lat,pred_lon) = full_pipeline(tensor, from_jpg=False, return_map=True)
    print(pred_lat, pred_lon)
    def latlon_to_pix(lat, lon):
        H, W = heatmap.shape
        row = int((90-lat) / 180 * H)
        col = int((lon+180) / 360 * W)
        return row, col
    nrows = 2
    ncols = max(2, len(preds))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10, 2*6))
    axs[0,0].imshow(image)
    axs[0,0].axis('off')

    im = axs[0,1].imshow(heatmap, cmap='viridis')
    cbar = plt.colorbar(im, ax=axs[0,1])
    if coords is not None:
        true_row, true_col = latlon_to_pix(true_lat, true_lon)
        axs[0,1].plot(true_col, true_row, 'ro', label='True', markersize=8)
    pred_row, pred_col = latlon_to_pix(pred_lat, pred_lon)
    axs[0,1].plot(pred_col, pred_row, 'bo', label='Pred', markersize=8)
    axs[0,1].legend()

    for c in range(2, len(preds)):
        axs[0,c].axis('off')
    for c in range(len(preds)):
        im = axs[1,c].imshow(preds[c], cmap='viridis')
        cbar = plt.colorbar(im, ax=axs[1,c])
    fig.suptitle("Full pipeline visualization")
    fig.savefig(f"./out/{fname}")

build_flickr_index()
dataset = GeoGoatDataset(flickr_transform, lambda x,y: (x,y))
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

climate_model = get_climate_model()
climate_model.load_state_dict(torch.load('./checkpoints/climate.pt', weights_only=True))
climate2_model = get_climate2_model()
climate2_model.load_state_dict(torch.load('./checkpoints/climate2.pt', weights_only=True))

index_to_country = dataset.index_to_country_map
num_countries = len(dataset.country_to_index_map)
print(f"num_countries: {num_countries}")
country_model = get_country_model(num_countries)
country_model.load_state_dict(torch.load('./checkpoints/country.pt', weights_only=True))

index_to_continent = dataset.index_to_continent_map
num_continents = len(dataset.continent_to_index_map)
print(f"num_continents: {num_continents}")
continent_model = get_continent_model(num_continents)
continent_model.load_state_dict(torch.load('./checkpoints/continent.pt', weights_only=True))

num_models = 5
def get_predictions(image, from_jpg=True):
    if from_jpg:
        image = flickr_transform(image).unsqueeze(0)
    preds = [
        climate_model_predict(climate_model, image, from_jpg=False),
        climate2_model_predict(climate2_model, image, from_jpg=False),
        country_model_predict(country_model, image, index_to_country, from_jpg=False)[0],
        continent_model_predict(continent_model, image, index_to_continent, from_jpg=False)[0],
        random_heatmap(),
    ]
    for i in range(len(preds)):
        preds[i] = preds[i] / preds[i].sum()
    return preds

def full_pipeline(image, from_jpg=True, return_map=False, weights=None):
    pred = get_predictions(image, from_jpg=from_jpg)
    if weights is None:
        weights = np.ones(len(pred)) / len(pred)
    heatmap = weights[0]*pred[0]
    for i in range(1, len(pred)):
        heatmap += weights[i]*pred[i]
    print("heatmap", heatmap, heatmap.shape)
    row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    print(row, col)
    transform = climate_src.transform
    lon, lat = rasterio.transform.xy(transform, row, col)
    if return_map:
        return heatmap, (lat, lon)
    return lat, lon

def visualize_all():
    for i in range(10):
        example = test_dataset[i]
        example_img = example[0].unsqueeze(0)
        example_coords = example[1]
        visualize(example_img, example_coords, fname=f'vis_pipeline/{i}.jpg')
    
if __name__ == '__main__':
    example = test_dataset[0]
    example_img = example[0].unsqueeze(0)
    example_coords = example[1]
    preds = get_predictions(example_img, from_jpg=False)
    print(preds)
    print(full_pipeline(example_img, from_jpg=False))

    visualize_all()

    random_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=100)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2, sampler=random_sampler)
    ensemble = GeoGoatEnsemble(len(preds))
    weights = ensemble.weights.clone().detach().cpu().numpy()
    optimizer = torch.optim.SGD(ensemble.parameters(), lr=5e-2, momentum=0)
    print(weights, weights.shape)
    def latlon_to_idx(lat, lon):
        H, W = 360, 720
        row = int((90-lat) / 180 * H)
        col = int((lon+180) / 360 * W)
        return row, col
    for images, coords in tqdm(train_dataloader):
        with torch.no_grad():
            preds = get_predictions(images, from_jpg=False)
            preds = tuple(map(lambda a: torch.tensor(a), preds))
            heatmaps = torch.stack(preds, dim=0)
        pred_heatmap = ensemble(heatmaps)
        assert pred_heatmap.shape == (360,720), pred_heatmap.shape
        row, col = latlon_to_idx(*coords)
        prob = pred_heatmap[row, col]
        loss = -torch.log(prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(ensemble.state_dict(), './checkpoints/ensemble.pt')
    weights = ensemble.weights.clone().detach().cpu()
    weights = F.softmax(weights, dim=0)
    weights = weights.numpy()
    print(weights, weights.shape)

    visualize_all()

