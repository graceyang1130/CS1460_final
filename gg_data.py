import msgpack
import geopandas as gpd
import shapely
import io
import os
import pwd
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

uname = pwd.getpwuid(os.getuid()).pw_name

geogoat_dir = f'/users/{uname}/scratch/GeoGoat/'
flickr_index_file = f'{geogoat_dir}indices/flickr.csv'

countries = gpd.read_file(f'{geogoat_dir}shapefiles/ne_110m_admin_0_countries.shp')
countries = countries.rename(columns={'ADMIN':'name'})

def open_shard(shard_n):
    with open(f'/users/{uname}/scratch/GeoGoat/shards/shard_0.msg', 'rb') as fb:
        unpacker = msgpack.Unpacker(fb)
        unpacked = list(unpacker)
    return unpacked

def build_flickr_index():
    if os.path.exists(flickr_index_file):
        print(f"Index already exists at {flickr_index_file}")
        return

    msgpack_files = sorted(glob(os.path.join(f'{geogoat_dir}shards', '*.msg')))
    if not msgpack_files:
        raise FileNotFoundError("No .msg files found")

    index_data = []
    for file in tqdm(msgpack_files, desc="Scanning Msgpack shards"):
        with open(file, 'rb') as f:
            unpacker = msgpack.Unpacker(f)
            for i, entry in enumerate(unpacker):
                lat = entry['latitude']
                lon = entry['longitude']
                img_file = entry['id'].replace('/','_')
                index_data.append({
                    'msgpack_file': os.path.basename(file),
                    'index_in_file': i,
                    'latitude': lat,
                    'longitude': lon,
                    'image': img_file
                })
                img = Image.open(io.BytesIO(entry['image'])).convert('RGB')
                img = img.resize((256, 256))
                img.save(os.path.join(f'{geogoat_dir}images', img_file), format='JPEG')

    # used GPT to help speed up the indexing through vectorizing get_country and get_continent
    points = gpd.GeoDataFrame(index_data, geometry=gpd.points_from_xy(
        [entry['longitude'] for entry in index_data],
        [entry['latitude'] for entry in index_data]
    ), crs='EPSG:4326')

    joined = gpd.sjoin(points, countries[['name', 'CONTINENT', 'geometry']], how='left', predicate='within')

    for i, row in enumerate(joined.itertuples()):
        index_data[i]['country'] = row.name
        index_data[i]['continent'] = row.CONTINENT

    df = pd.DataFrame(index_data)
    df.to_csv(flickr_index_file, index=False)
    print(f"Index saved to {flickr_index_file} ({len(df)} samples)")

flickr_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

class GeoGoatDataset(Dataset):
    def __init__(self, transform, get_label):
        self.shards_dir = f"/users/{uname}/scratch/GeoGoat/shards/"
        self.transform = transform
        self.get_label = get_label
        self.index_df = pd.read_csv(flickr_index_file)
        self.countries = self.index_df['country'].dropna().unique()
        self.country_to_index_map = {country: i for i, country in enumerate(self.countries)}
        self.index_to_country_map = {i: country for i, country in enumerate(self.countries)}
        self.continents = self.index_df['continent'].dropna().unique()
        self.continent_to_index_map = {continent: i for i, continent in enumerate(self.continents)}
        self.index_to_continent_map = {i: continent for i, continent in enumerate(self.continents)}
        print(f"Total samples: {len(self)}")

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        msgpack_file = os.path.join(self.shards_dir, row['msgpack_file'])
        entry_idx = row['index_in_file']
        lat = row['latitude']
        lon = row['longitude']
        image_dir = os.path.join(geogoat_dir, 'images')
        image_file = os.path.join(image_dir, row['image'])
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        # climate = coords_to_climate(lat, lon)
        # return image, climate
        # label = self.country_to_index_map[self.get_label(lat, lon)]
        if self.get_label.__name__ == "get_country":
            label = self.country_to_index_map[self.get_label(lat, lon)]
        elif self.get_label.__name__ == "get_continent":
            label = self.continent_to_index_map[self.get_label(lat, lon)]
        else:
            label = self.get_label(lat, lon)
        return image, label

def split_dataset(full_dataset):
    torch.manual_seed(42)
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.1, 0.1])
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(full_dataset, [0.16, 0.02, 0.02, 0.8])
    return train_dataset, val_dataset, test_dataset

def get_country(lat, long):
    point = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(long,lat)], crs='EPSG:4326')
    joined = gpd.sjoin(point, countries, how='left', predicate='within')
    country = joined.iloc[0]['name']
    return country

def get_continent(lat, long):
    point = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(long, lat)], crs='EPSG:4326')
    joined = gpd.sjoin(point, countries, how='left', predicate='within')
    return joined.iloc[0]['CONTINENT']
