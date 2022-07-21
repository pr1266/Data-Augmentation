from cv2 import cv2
from matplotlib import pyplot as plt
import albumentations as A
import glob
import os
from custom_functional_transforms import *
import torchvision.transforms.functional as my_f
from torchvision.transforms import Compose
from torchvision import transforms

os.system('cls')

img_path = 'teeth_data/279.jpg'
txt_path = 'teeth_data/279.txt'

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



lines = []
with open(txt_path, 'r') as F:
    lines = F.readlines()

bbox = []
for line in lines:
    line = line.replace('\n', '')
    elements = line.split(',')
    single_box = [float(i) for i in elements[1:]]
    single_box.append(elements[0])
    bbox.append(single_box)

transform = A.Compose([
    A.RandomCrop(width=800, height=800),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo'))


transformed = transform(image=image, bboxes=bbox)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
new_image = transformed_image
dh, dw, _ = new_image.shape
for box in transformed_bboxes:
    print(box)
    x, y, w, h, c = box

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cropped = new_image[t:b,l:r]
    tr = Compose([
            transforms.ToPILImage(),
            CustomTransform(my_f.adjust_contrast, 3.0),
            # transforms.ToTensor()
        ])
    new_cropped = tr(cropped)
    new_image[t:b,l:r] = new_cropped

    color = (255, 0, 0) if c == 'broken' else (0, 255, 0) 
    cv2.rectangle(new_image, (l, t), (r, b), color, 6)

plt.imshow(new_image)
plt.show()