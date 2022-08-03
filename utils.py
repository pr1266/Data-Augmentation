
import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import glob
import random
import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def convert_yolo(size, box):
    #! yolo format: [x_center, y_center, width, height] and all values are normalized
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_coco(size, box, resize):
    #! coco format: [x_min, y_min, width, height] values are not normalized
    w_ratio = float(resize[0]) / size[0]
    h_ratio = float(resize[1]) / size[1]
    x_min = box[0] * w_ratio
    y_min = box[1] * h_ratio
    x_max = box[2] * w_ratio
    y_max = box[3] * h_ratio
    return (x_min, y_min, x_max, y_max)

def convert_xml_annotation(filename, coord_type, resize, task='object detection', name_file=None):
    #! pascal format: [x_min, y_min, x_max, y_max] values are not normalized
    with open(filename) as in_file:
        filename, file_extension = os.path.splitext(filename)
        tree = ET.parse(in_file)
        root = tree.getroot()
        object_ = root.find('object')
        #! if a xml is empty, just skip it.
        if object_ is None:
            in_file.close()
            print('WARNING: There is no object in the annotation file {}.xml. The observation is ignored.'.format(filename))
            return
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if width <= 0 or height <= 0:
            in_file.close()
            print('WARNING: Please check the annotation file, {}.xml, '
                  'in which either width or height is smaller than 0. The observation is ignored.'.format(filename))
            return
        if name_file:
            with open(name_file, 'r') as nf:
                line = nf.readlines()
                cls_list = [l.strip() for l in line]

        with open(filename + ".txt", 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls != 'ignore':
                    xmlbox = obj.find('bndbox')
                    xmin = float(xmlbox.find('xmin').text)
                    xmax = float(xmlbox.find('xmax').text)
                    ymin = float(xmlbox.find('ymin').text)
                    ymax = float(xmlbox.find('ymax').text)
                    if xmin >= xmax or ymin >= ymax:
                        print('WARNING: Please check the annotation file, {}.xml, '
                              'which contains an invalid bounding box.'.format(filename))
                    boxes = (xmin, ymin, xmax, ymax)

                    if coord_type == 'yolo':
                        boxes = convert_yolo((width, height), boxes)

                    elif coord_type == 'coco':
                        boxes = convert_coco((width, height), boxes, resize)

                    if name_file:
                        try:
                            idx = str(cls_list.index(str(cls)))

                        except ValueError:
                            raise print('{} is not in name file'.format(str(cls)))
                        out_file.write(idx + " " + " ".join([str(box) for box in boxes]) + '\n')                    
                    else:
                        
                        out_file.write(str(cls) + " " + " ".join([str(box) for box in boxes]) + '\n')


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    dh, dw, _ = img.shape
    x_min, y_min, w, h, _ = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
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
        
    cv2.rectangle(img, (l, t), (r, b), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)  
    cv2.rectangle(img, (l, t - int(1.3 * text_height)), (l + text_width, t), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(l, t - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    # plt.imshow(img)
    # plt.show()
    return transforms.ToTensor()(img)

def load_bbox(txt_path):

    lines = []
    with open(txt_path, 'r') as F:
        lines = F.readlines()

    bbox = []
    for line in lines:
        line = line.replace('\n', '')
        elements = line.split(' ')
        single_box = [float(i) for i in elements[1:]]
        single_box.append(elements[0])
        bbox.append(single_box)
    return bbox

def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def test():

    # image = cv2.imread('test_data/00inner_bbox.jpg')
    # bbox = load_bbox('test_data/00inner_bbox.txt')
    # visualize(image, bbox, [0], {0:'raccoon'})
    
    os.system('cls')
    list = glob.glob('test_data/*.jpg')
    list = random.sample(list, 36)
    bbox = [load_bbox(i[:-3]+'txt') for i in list]
    
    list = [cv2.imread(i) for i in list]
    image_list = [visualize(list[i], bbox[i], [0], {0:'raccoon'}) for i in range(len(list))]
    grid = make_grid(image_list, nrow=6)
    save_image(grid, 'final.jpg')

if __name__ == '__main__':
    test()

