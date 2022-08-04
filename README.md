# Expand your Dataset to infinite!
One of the most common issues that data scientists are faced with in real AI and Computer Vision projects is data insufficiency. Deep Learning algorithms usually need a lot of data to solve our problem and data gathering is expensive, time-consuming, and in some cases impossible, therefore, data augmentation is an important task to generate massive data from a small dataset
In this project, you can apply many augmentation methods to your data to generate a massive and sufficient dataset
from your small one.

### Built With
* [![PyTorch][torchlogo]][torchurl]
* [![Albumentation][alblogo]][alburl]

## About Project
![alt text](https://github.com/pr1266/data_augmentation/blob/master/src/final.jpg)

# how does this project help you?
we designed it to performing image augmentation for:
* Normal Classification
* Object Detection
* Semantic Segmentation (soon)
* Keypoint Detection (soon)

for different backbones witht different input size you can set the output size according to your desired architecture 

```python
cfg = {
    'format': 'yolo',
    'target_size': (640, 640),
    'bounding_box': [
        A.CenterCrop(100, 100),
        A.RandomCrop(100, 100),
        CustomTransform(F.adjust_brightness, 3.0),
        CustomTransform(F.adjust_contrast, 4.2),
        CustomTransform(F.adjust_sharpness, 3.0),
        transforms.Grayscale(),
        CustomTransform(my_f.adjust_saturation, 8),
        CustomTransform(F.adjust_hue, -0.3),
        CustomGaussianBlurTransform(None, 5),
    ],
    'inner_bounding_box': [
        transforms.RandomEqualize(1.0),
        CustomTransform(F.adjust_brightness, 3.0),
        CustomTransform(F.adjust_contrast, 4.2),
        CustomTransform(F.adjust_sharpness, 3.0),
        transforms.Grayscale(),
        CustomTransform(my_f.adjust_saturation, 8),
        CustomTransform(F.adjust_hue, -0.3),
        CustomGaussianBlurTransform(None, 5),
    ]
}
```



[torchlogo]: https://img.shields.io/badge/pytorch-ff8200?style=for-the-badge&logo=PyTorch&logoColor=white
[torchurl]: https://pytorch.org/

[alblogo]: https://img.shields.io/badge/Albumentations-FFFFFF?style=for-the-badge
[alburl]: https://albumentations.ai/
