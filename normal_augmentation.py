import torch
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from cv2 import cv2

torch.manual_seed(17)
os.system('cls')

#! first we specify path to our data:
data_dir = 'data'
#! and then create a transform compose object to perform some sequential transformations like Resize, Crop and etc. 
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
    ])

#! then create a dataset object
dataset = datasets.ImageFolder(data_dir, transform=transform)
print(dataset)
data_loader = DataLoader(dataset, batch_size=5, shuffle=False)
data_loader = iter(data_loader)

img, label = next(data_loader)

# print the total no of samples
print('Number of samples: ', len(img))
image = img[2][0]  # load 3rd sample
  
# visualize the image
plt.imshow(image)
plt.show()
