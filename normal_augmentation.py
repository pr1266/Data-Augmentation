import torch
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

os.system('cls')

#! first we specify path to our data:
data_dir = 'data'
#! and then create a transform compose object to perform some sequential transformations like Resize, Crop and etc. 
transform = Compose([
    transforms.Resize(255), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()])

#! then create a dataset object
dataset = datasets.ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=5, shuffle=False)
data_loader = iter(DataLoader)

img, label = next(data_loader)

# print the total no of samples
print('Number of samples: ', len(img))
image = img[2][0]  # load 3rd sample
  
# visualize the image
plt.imshow(image)
  
# print the size of image
print("Image Size: ", image.size())
  
# print the label
print(label)
