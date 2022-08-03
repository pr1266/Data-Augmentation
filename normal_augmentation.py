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
from custom_functional_transforms import CustomGaussianBlurTransform, CustomTransform

torch.manual_seed(17)
os.system('cls')

#! first we specify path to our data:
data_dir = 'raccoon'
size = (255, 255)

config = {
    '1st_stage_augment': [
        transforms.Grayscale(),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '2nd_stage_augment': [
        transforms.GaussianBlur((7, 7)),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '3rd_stage_augment': [
        transforms.RandomEqualize(1.0),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '4th_stage_augment': [
        CustomTransform(F.adjust_brightness, 3.0),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '5th_stage_augment': [
        CustomTransform(F.adjust_contrast, 4.2),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '6th_stage_augment': [
        CustomTransform(F.adjust_hue, -0.3),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '7th_stage_augment': [
        CustomTransform(F.adjust_saturation, 7.0),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '8th_stage_augment': [
        CustomTransform(F.adjust_sharpness, 3.0),
        transforms.Resize(size),
        transforms.ToTensor()
    ],
    '9th_stage_augment': [
        CustomGaussianBlurTransform(None, 7),
        transforms.Resize(size),
        transforms.ToTensor()
    ]
}

for index, key in enumerate(config):

    dataset = datasets.ImageFolder(data_dir, transform=Compose(config[key]))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_loader = iter(data_loader)
    index = 0

    if not os.path.exists(key):
        os.makedirs(key)

    for img, label in data_loader:
        # grid = torchvision.utils.make_grid(img)
        # img = torchvision.transforms.ToPILImage()(img)
        path = f'{key}/'+str(index)+'.jpg'
        torchvision.utils.save_image(img, path)
        index += 1