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
data_dir = 'data'


config = {
    '1st_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ],
    '2nd_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.GaussianBlur((7, 7)),
        transforms.ToTensor()
    ],
    '3rd_stage_augment': [
        transforms.Resize((255, 255)),
        transforms.RandomEqualize(1.0),
        transforms.ToTensor()
    ],
    '4th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomTransform(F.adjust_brightness, 3.0),
        transforms.ToTensor()
    ],
    '5th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomTransform(F.adjust_contrast, 4.2),
        transforms.ToTensor()
    ],
    '6th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomTransform(F.adjust_hue, -0.3),
        transforms.ToTensor()
    ],
    '7th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomTransform(F.adjust_saturation, 7.0),
        transforms.ToTensor()
    ],
    '8th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomTransform(F.adjust_sharpness, 3.0),
        transforms.ToTensor()
    ],
    '9th_stage_augment': [
        transforms.Resize((255, 255)),
        CustomGaussianBlurTransform(None, 7),
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