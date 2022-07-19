import torch
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

torch.manual_seed(17)
os.system('cls')

#! first we specify path to our data:
data_dir = 'data'


config = {
    'first_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ],
    'second_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.GaussianBlur((7, 7)),
        transforms.ToTensor()
    ],
    'third_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.RandomEqualize(1.0),
        transforms.ToTensor()
    ],
    'fourth_stage_augment': [
        transforms.Resize((255, 255)),
        AdjustBrighnessTransform(3.0),
        transforms.ToTensor()
    ]
}

for index, key in enumerate(config):

    dataset = datasets.ImageFolder(data_dir, transform=Compose(config[key]))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    data_loader = iter(data_loader)
    index = 0

    if not os.path.exists(key):
        os.makedirs(key)

    for img, label in data_loader:
        image = img
        grid = torchvision.utils.make_grid(img)
        img = torchvision.transforms.ToPILImage()(grid)
        path = f'{key}/'+str(index)+'.jpg'
        torchvision.utils.save_image(image, path)
        index += 1