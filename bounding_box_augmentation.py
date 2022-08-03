from cv2 import cv2
from matplotlib import pyplot as plt
import albumentations as A
import glob
import os
from custom_functional_transforms import *
import torchvision.transforms.functional as my_f
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import Dataset
from utils import *
from torchvision.utils import save_image

os.system('cls')

data_dir = 'teeth_data/'
classes = {
    'normal': 0,
    'broken': 1
}
cfg = {
    'format': 'yolo',
    'target_size': (400, 400),
    'bounding_box': [
        A.CenterCrop(800, 800),
        A.RandomCrop(800, 800),
        A.VerticalFlip(True, 1.0)
    ],
    'inner_bounding_box': [
        CustomTransform(my_f.adjust_saturation, 8),
        CustomGaussianBlurTransform(None, 20),
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
            transformed = transform(image=img, bboxes=bbox)
            transformed_image = transformed['image']                
            transformed_bboxs = transformed['bboxes']
            image_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'bbox' + '.jpg'
            bbox_save_path = self.save_dir + '/' + str(index) + str(self.index) + 'bbox' + self.prefix
            image_to_save = transforms.ToTensor()(transformed_image)
            save_image(image_to_save, image_save_path)
            with open(bbox_save_path, 'w') as f:
                #! just implemented for yolo format because im currently use it
                #! other formats will be added soon
                for index, i in enumerate(transformed_bboxs):             
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
                t = A.Compose([
                    augmentation,
                    A.Resize(self.target_size[0], self.target_size[1])
                ],
                bbox_params=A.BboxParams(format=self.format))
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
            for bbox in bboxs:
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

                cropped = img[t:b,l:r]
                # tr = Compose([
                #     CustomTransform(my_f.adjust_saturation, 8),
                # ])
                new_cropped = transform(cropped)
                img[t:b,l:r] = new_cropped
            
            image_save_path = self.save_dir + '/' + str(index) + str(self.index) + '.jpg'
            bbox_save_path = self.save_dir + '/' + str(index) + str(self.index) + self.prefix

            image_to_save = transforms.ToTensor()(transformed_image)
            #! resize whole image to target size:
            image_to_save = transform.Resize()(image_to_save)
            
            save_image(image_to_save, image_save_path)
            with open(bbox_save_path, 'w') as f:
                #! just implemented for yolo format because im currently use it
                #! other formats will be added soon
                for index, i in enumerate(transformed_bboxs):             
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
                t = A.Compose([
                    augmentation,
                ],
                bbox_params=A.BboxParams(format=self.format))
                _transform.append(t)
        return _transform

def main():
    dataset = MyDataset('teeth_data/', 'yolo')
    bbox_augmentation = BoundingBoxAugmentation(cfg)
    for i in range(len(dataset)):
        bbox_augmentation(dataset[i])

if __name__ == '__main__':
    main()

# transform = A.Compose(config['1st_stage']['bounding_box'], bbox_params=A.BboxParams(format='yolo'))
# transformed = transform(image=image, bboxes=bbox)
# transformed_image = transformed['image']
# transformed_bboxes = transformed['bboxes']
# new_image = transformed_image
# dh, dw, _ = new_image.shape
# for box in transformed_bboxes:
#     x, y, w, h, c = box

#     l = int((x - w / 2) * dw)
#     r = int((x + w / 2) * dw)
#     t = int((y - h / 2) * dh)
#     b = int((y + h / 2) * dh)
    
#     if l < 0:
#         l = 0
#     if r > dw - 1:
#         r = dw - 1
#     if t < 0:
#         t = 0
#     if b > dh - 1:
#         b = dh - 1

#     cropped = new_image[t:b,l:r]
#     tr = Compose([
#             # transforms.ToPILImage(),
#             CustomTransform(my_f.adjust_saturation, 8),
#         ])
#     new_cropped = tr(cropped)
#     new_image[t:b,l:r] = new_cropped

#     color = (255, 0, 0) if c == 'broken' else (0, 255, 0) 
#     cv2.rectangle(new_image, (l, t), (r, b), color, 6)

# plt.imshow(new_image)
# plt.show()