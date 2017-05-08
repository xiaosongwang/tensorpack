
## ChestX-ray8:  Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases

Code of our CVPR 2017 paper. See .

![ChestX-ray8](demo.jpg)

## Usage

This script only needs the original ChestX-ray8 dataset.

It requires ImageNet pretrained vgg16/GoogLeNet/ResNet model. See the docs in [examples/load-vgg16.py](../load-vgg16.py)
for instructions to convert from vgg16 caffe model.

To view augmented training images:
python multilabel.py --display


To start training:
python multilabel.py --train vgg16.npy

To inference (produce a heatmap at each level at out*.png):
python multilabel.py --train pretrained.model --test a.jpg

