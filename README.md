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
for object detection tasks with bounding boxes, you can perform both bounding-box and inner-bounding-box augmentation
if you want to add spatial-level augmentation like crop, rotate, padding or flip, you must add it through Albumentation and pass the bboxes
and formats to it
and also you can convert Pascal-VOC format to your ideal format like YOLO and COCO using convert functions implemented in utils.py

```python
import albumentations as A

t = A.Compose([
        augmentation,
        A.Resize(width, height)
    ],
    bbox_params=A.BboxParams(format=self.format)) #for example yolo
```

in case of using torchvision functional transforms, you must create a CustomTransform instance and pass that
functional transformer to it. (implemented in detail in custom_functional_transformers)

## Some Interesting Results !
![alt text](https://github.com/pr1266/data_augmentation/blob/master/src/final_res.jpg)

## Contact

* gmail: pour.pourya1999@gmail.com
* [![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/pr1266/)

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

[torchlogo]: https://img.shields.io/badge/pytorch-ff8200?style=for-the-badge&logo=PyTorch&logoColor=white
[torchurl]: https://pytorch.org/

[alblogo]: https://img.shields.io/badge/Albumentations-FFFFFF?style=for-the-badge
[alburl]: https://albumentations.ai/
