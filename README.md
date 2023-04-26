# ConstantinovMihai.github.io

Reproducibility project for the Deep Learning course. This repository contains the blogpost we wrote, as well as the code used
to reproduced the results of the original EfficientNetV2 paper. 

The blogpost can be found in the folder *posts* under the name *2023-04-13-EfficientNetV2.md*.

The code used can be find under the folder *code*, and it contains the following files:

* *Implementation-ImageNetTE.py* - This file contains the PyTorch implementation such as described in the blog post which trains on ImageNetTE. When running this file, first the training images for ImageNetTE need to be downloaded from https://github.com/fastai/imagenette (160px download) and need to be added in a "data" folder located in the same folder as this file.

* *Implementation-FashionMNIST.ipynb* - This file contains the implementation of the FashionMNIST dataset in Pytroch, similarly described in the blogpost. The dataset should also be downloaded into the same directory and added into a "data" folder.

The attempts made in tensorflow were made in the original code provided under https://github.com/google/automl/tree/master/efficientnetv2. All changes are explained and shown as code blocks in the blog post.
