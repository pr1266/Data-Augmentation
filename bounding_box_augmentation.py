from cv2 import cv2
from matplotlib import pyplot as plt
import albumentations as A
import glob
import os
import PIL
from numpy import isin
from custom_functional_transforms import *
import custom_functional_transforms
import torchvision.transforms.functional as my_f
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import Dataset
from utils import *
from torchvision.utils import save_image
"""
one of the most common issues that data scientist are faced in real AI and Computer Vision
projects is data insufficiency. Deep Learning algorithms usually needs a lot of data to solve our problem
and data gathering is expensive, time-consuming, and in some cases impossible
therefore, data augmentation is an important task to generate massive data from a small dataset
in this project, you can apply many augmentation methods to your data to generate a massive and sufficient dataset
from your samll one. just change data_dir to your txt-jpg containing directory and run this script.
"""
os.system('cls')

data_dir = 'raccoon/raccoon'
classes = {
    'raccoon': 0,
}

#! you can find more augmentation methods on PyTorch and Albumentation Docs
cfg = {
    'format': 'yolo',
    'target_size': (640, 640),
    'bounding_box': [
        # A.CenterCrop(100, 100),
        # A.RandomCrop(100, 100),
        CustomTransform(F.adjust_brightness, 3.0),
        CustomTransform(F.adjust_contrast, 4.2),
        CustomTransform(F.adjust_sharpness, 3.0),
        # transforms.Grayscale(),
        CustomTransform(my_f.adjust_saturation, 8),
        CustomTransform(F.adjust_hue, -0.3),
        CustomGaussianBlurTransform(None, 5),
    ],
    'inner_bounding_box': [
        transforms.RandomEqualize(1.0),
        CustomTransform(F.adjust_brightness, 3.0),
        CustomTransform(F.adjust_contrast, 4.2),
        CustomTransform(F.adjust_sharpness, 3.0),
        # transforms.Grayscale(),
        CustomTransform(my_f.adjust_saturation, 8),
        CustomTransform(F.adjust_hue, -0.3),
        CustomGaussianBlurTransform(None, 5),
    ]
}

class MyDataset(Dataset):

    def __init__(self, path, format):
        assert format in ['yolo', 'coco', 'xml'], "format must be in yolo, coco or pascal"
        self.path = path
        self.format = format
        self.data = []
        #! inja bekhoon:
        list = glob.glob(self.path+'*.jpg')
        
        self.prefix = 'txt'
        if self.format == 'yolo':
            self.prefix = 'txt'
        elif self.format == 'pascal':
            self.prefix = 'xml'
        elif self.format == 'coco':
            self.prefix = 'json'

        for i in list:
            image = cv2.imread(i)
            bbox = load_bbox(i[:-3]+self.prefix)
            dict_ = {
                'image': image,
                'bbox': bbox
            }
            self.data.append(dict_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class BoundingBoxAugmentation:

    def __init__(self, cfg, save_dir='test_data'):
        self.index = 0
        self.format = cfg['format']
        self.prefix = None
        self.delimiter = None

        if self.format == 'yolo':
            self.prefix = '.txt'
            self.delimiter = ' '

        elif self.format == 'coco':
            self.prefix = '.json'
            #! soon

        elif self.format == 'pascal':
            self.prefix = '.xml'
            #! soon

        self.target_size = cfg['target_size']
        self.cfg = cfg
        self.transforms = self.create_transform()
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, data):
        index=0
        img = data['image']
        bbox = data['bbox']
        
        for transform in self.transforms:
            new_img = img
            
            if isinstance(transform, transforms.Compose):
                transformed_image = transform(new_img)
                transformed_bboxs = bbox

            else:
                transformed = transform(image=new_img, bboxes=bbox)
                transformed_image = transformed['image']
                transformed_bboxs = transformed['bboxes']

            image_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'bbox' + '.jpg'
            bbox_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'bbox' + self.prefix
            image_to_save = transforms.ToTensor()(transformed_image)
            save_image(image_to_save, image_save_path)
            with open(bbox_save_path, 'w') as f:
                #! just implemented for yolo format because im currently use it
                #! other formats will be added soon
                for _, i in enumerate(transformed_bboxs):             
                    vals = [str(val) for val in list(i[:-1])]                    
                    to_write = str(classes[i[-1]]) + self.delimiter + self.delimiter.join(vals) + '\n'                    
                    f.writelines(to_write)
            index += 1
        self.index += 1

    def create_transform(self):

        cfg = self.cfg
        _transform = []
        if 'bounding_box' in self.cfg.keys():
            for augmentation in cfg['bounding_box']:
                t = None
                if type(augmentation) != custom_functional_transforms.CustomTransform and type(augmentation) != custom_functional_transforms.CustomGaussianBlurTransform:
                    t = A.Compose([
                        augmentation,
                        A.Resize(self.target_size[0], self.target_size[1])
                    ],
                    bbox_params=A.BboxParams(format=self.format))
                else:
                    t = transforms.Compose([
                        augmentation,
                        transforms.Resize((self.target_size[0], self.target_size[1]))
                    ])
                _transform.append(t)
        return _transform

class InnerBoundingBoxAugmentation:
    def __init__(self, cfg, save_dir='test_data'):
        self.index = 0
        self.format = cfg['format']
        self.prefix = None
        
        if self.format == 'yolo':
            self.prefix = '.txt'
        elif self.format == 'coco':
            self.prefix = '.json'
        elif self.format == 'pascal':
            self.prefix = '.xml'

        self.target_size = cfg['target_size']
        self.cfg = cfg
        self.transforms = self.create_transform()
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, data):
        index = 0
        img = data['image']
        bboxs = data['bbox']
        dh, dw, _ = img.shape

        for transform in self.transforms:
            new_image = img
            new_bboxs = bboxs
            for bbox in new_bboxs:
                x, y, w, h, c = bbox
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
                if isinstance(cropped, PIL.Image.Image):
                    new_cropped = transform(cropped)
                else:
                    new_cropped = transform(transforms.ToPILImage()(cropped))
                new_image[t:b,l:r] = new_cropped

            image_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'inner_bbox' + '.jpg'            
            bbox_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'inner_bbox' + self.prefix
            

            image_to_save = transforms.ToTensor()(new_image)
            #! resize whole image to target size:
            image_to_save = transforms.Resize(self.target_size)(image_to_save)
            
            save_image(image_to_save, image_save_path)
            with open(bbox_save_path, 'w') as f:
                #! just implemented for yolo format because im currently use it
                #! other formats will be added soon
                for _, i in enumerate(bboxs):  
                    vals = [str(val) for val in list(i[:-1])]                    
                    to_write = i[-1] + ' ' + ' '.join(vals)+'\n'                    
                    f.writelines(to_write)
            index += 1
        self.index += 1

    def create_transform(self):

        cfg = self.cfg
        _transform = []
        if 'inner_bounding_box' in self.cfg.keys():
            for augmentation in cfg['inner_bounding_box']:
                #! to change:
                #! in the inner bounding box augmentation
                #! we not use the last resize augmentation method
                #! why? because we are going to change inner bbox content not the whole image
                #! and in the end we apply resize augmentation on whole image but not bbox content 
                t = Compose([
                    augmentation,
                ])
                _transform.append(t)
        return _transform

def inner_box_augmentation():
    dataset = MyDataset('raccoon/raccoon/', 'yolo')
    inner_bbox_augmentation = InnerBoundingBoxAugmentation(cfg)
    for i in range(len(dataset)):
        inner_bbox_augmentation(dataset[i])

def bounding_box_augmentation():
    dataset = MyDataset('raccoon/raccoon/', 'yolo')
    bbox_augmentation = BoundingBoxAugmentation(cfg)
    for i in range(len(dataset)):
        bbox_augmentation(dataset[i])

def main():
    inner_box_augmentation()
    bounding_box_augmentation()

    


if __name__ == '__main__':
    main()