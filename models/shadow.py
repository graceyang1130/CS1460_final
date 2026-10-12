import os
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gg_data import *

# copied from climate.py
def dls(full_dataset, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataset

def extract_shadow_angle(img):
    img = np.array(img.convert('L'))
    # grayscale, then blur ->>> cannny for edge + hough line tansform to get the lines
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return None
    angles = []
    # getting the average orientation of the shadow from the center
    for _, theta in lines[:, 0]:
        angle = math.degrees(theta) % 180 # wrap
        angles.append(angle)
    return np.median(angles)

#import logging # use tqdm

#inspiration from the GeoGoatDataset class for this one, they use tqdm
class shadowDataset(Dataset):
    def __init__(self, index_df, image_dir):
        self.image_dir = image_dir
        valid_rows = []
        missing_count = 0
        for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Ading images, calculating shadows, etc.."):
            img_path = os.path.join(image_dir, row['image'])
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                missing_count += 1
        self.index_df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        lat = abs(row['latitude'])  # absolute distance from equator
        img_path = os.path.join(self.image_dir, row['image'])
        img = Image.open(img_path).convert('RGB')
        # img = Image.new('RGB', (224, 224), color='black')  # fallback
        angle = extract_shadow_angle(img)
        if angle is None:
            angle = 0.0
        return torch.tensor([angle], dtype=torch.float32), torch.tensor([lat], dtype=torch.float32)

# normal ResNet will probably not work well here
# decided to create a standard regressive nn, two values are easy to relate (i.e. orientation and abs lat)
class mdl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
geogoat_dir = f'/users/{uname}/scratch/GeoGoat/'
flickr_index_file = f'{geogoat_dir}indices/flickr.csv'

if __name__ == "__main__":
    # from climate.py
    print("Running")

    print("Loading index")
    sys.stdout.flush()
    index_df = pd.read_csv(flickr_index_file)
    index_df = index_df.dropna(subset=['latitude'])

    print("Building dataset")
    sys.stdout.flush()
    dataset = shadowDataset(index_df, image_dir=os.path.join(geogoat_dir, "images"))
    traindl, valdl, test_dataset = dls(dataset, 64, 1)

    model = mdl().cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    min_val_loss = None

    # copied from shared.py
    for epoch in range(5):
        running_loss = 0.0
        pbar = tqdm(traindl)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"train loss: {running_loss/(i+1)}")
        train_loss = running_loss / len(traindl)

        running_loss = 0.0
        total = 0
        correct = 0
        pbar = tqdm(valdl)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_description(f"val loss: {running_loss/(i+1)}, acc: {correct/total}")
        val_loss = running_loss / len(valdl)
        acc = correct / total
        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoints/shadow_latitude.pt')

        print(f"Epoch {epoch+1}: train loss={train_loss}, val_loss={val_loss}, acc={acc}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), "./checkpoints/shadow_latitude.pt")
    print("Model saved to ./checkpoints/shadow_latitude.pt")


