# EfficientNetV2 Reproducibility project

## Introduction

## Reproducibility Tensorflow Implementation

### Description of Original Tensorflow Implementation

### Implementation of a New Dataset

Several datasets are already implemented in the tensorflow implementation of the EfficientNetV2 model. The datasets implemented are: ImageNet, ImageNet21k, ImageNetTfds, Cifar10, Cifar100, Flowers, TFFlowers and Cars. In order to analyze if the implementation of EfficientNetV2 is adaptable to other datasets as well, an attempt was made to introduce a new dataset: FashionMNIST [[1]](#1). FashionMNIST is a dataset containing 10 different classes of Zalando product types, with in total 60000 testing and 10000 validation examples. They are all provided as grayscaled images with dimensions of 28x28 pixels.

In order to implement a new type of dataset, several changes needed to be made to the file *datasets.py*. First of all, a new class needed to be created for the new input of FashionMNIST. Similarly, to the other datasets, the new class was created by copying an older class and only changing the relevant entries in the dictionary. The new implementation was based on the class CIFAR10Input, which was the class used in the tutorial and thus (in theory) proven to be working. In the configuration dictionary of the new FashionMNIST class, the following elements were updated:

* The number of classes was set to 10.
* The dataset name was set to *fashion_mnist*, which is already implemented in tensorflow.
* The number of training images was set to 60000.
* The number of validation images was set to 10000 (minival, eval).

The implementation of the new class in the code can be seen below.

```ruby
class FashionMNISTInput(CIFAR10Input):
  """Generates input_fn from FashionMNIST files."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          num_classes=10,
          tfds_name='fashion_mnist',
          splits=dict(
              train=dict(num_images=60000, tfds_split='train', slice=''),
              minival=dict(num_images=10000, tfds_split='test', slice=''),
              eval=dict(num_images=10000, tfds_split='test', slice=''),
          )))
```

Next, the FashionMNIST dataset also needed to be added to the function *get_dataset_class(ds_name)*, which is calling the class of a dataset in the main file of EfficientNetV2. The entry was added in the same way as for all other datasets by linking the name of the dataset (*fashion_mnist*) to the name of the previously defined new class (*FashionMNISTInput*). The entry added to the function was therefore as shown below.

```ruby
'fashion_mnist': FashionMNISTInput,
```

Finally, similarly to the other datasets a training configuration needed to be registered. This was again done by copying the training configuration *Cifar10Ft.cfg* of Cifar10 to the new class *FashionMNISTFt* for the configuration of FashionMNIST. The configuration data was overwritten with the data which was previously defined in the class of FashionMNIST. This was done as follows.

```ruby
@ds_register
class FashionMNISTFt(Cifar10Ft):
  """Finetune fashionmnist configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='fashion_mnist'))
```

As the dataset of FashionMNIST is already implemented in tensorflow, those changes should be sufficient in order to implement this new dataset for EfficientNetV2. The old file *datasets.py* was thus then replaced by the new file containing the new implementations for FashionMNIST. 

### Possible Ablation Study

Next, an ablation study was supposed to be performed. As introduced in the article, an improvement in this new model was the progressive learning using an adaptive regularization. The image size was adpted in line 468 to 475 of the file *main.py*. To be very precise, the computations required were performed in line 472 to 474 and are shown as follows [[2]](#2).

```ruby
ratio = float(stage + 1) / float(total_stages)
max_steps = int(ratio * train_steps)
image_size = int(ibase + (input_image_size - ibase) * ratio)
```

The idea was to compare in the ablation study, how training would have performed when fully removing progressive learning with an adaptive regularization. Therefore, the code block shown above would need to be replaced with the one following, in order to have a constant image size. 

```ruby
image_size = input_image_size
```

Next to that, also the adaptive regularization would need to be removed. Regularization was found to also be performed in the file *main.py* from line 126 to 142. Again, some lines in the code were found to be most relevant for the regularization, namely line 129 to 131 defining the weight_decay as follows [[2]](#2).

```ruby
weight_decay_inc = config.train.weight_decay_inc * (
    tf.cast(global_step, tf.float32) / tf.cast(train_steps, tf.float32))
weight_decay = (1 + weight_decay_inc) * config.train.weight_decay
```

Similaraly to the image size, a new version of those code lines would need to be implemented. The new line of code replacing the lines above and defining the weight decay in a different way in order to remove adaptive regularization would have been the following.

```ruby
weight_decay = 1
```

With those two changes, it would be possible to test how training would perform without using adaptive regularization for the progressive learning. The difficulties faced while attempting to train the model, which prevented running the ablation study and the implementation of a new dataset, are described in the following subsection.

### Problems With Tensorflow Implementation

As was described in the previous subsection, in this reproducibility project an attempt was made to reproduce the results presented in the paper on the model EfficientNetV2 [[1]](#1), as well as to try implementing a new dataset as well as performing an ablation study by removing adaptive regularization. However, unfortunately, several issues were faced while trying to run the code provided in the paper. They are summarized as follows.

In the repository provided (a link to the GitHub repositroy was given in the paper [[1]](#1)), a tutorial was given on how to run the different types of the EfficientNet model with the files provided in the repository. In this tutorial, the model EfficientNet-B0 is finetuned. In a first step, it was therefore tried to run this tutorial. When strictly running what was given in the tutorial, training of the model was initialized and started running. Special attention was given to the Top-1 and Top-5 accuracy. In the first 56 of 781 training epochs, the Top-1 accuracy was observed to stay below 10% and the Top-5 accuracy varied between 40% and 50%, but did not exceed the 50%. It even decreased again for multiple of the epochs. 

Unfortunately, after running the code for nearly 1 day, it crashed. The computing power of a single computer were seen to not be sufficient to train the model in a reasonable timeframe. Attempts to run it in the Google Cloud were unfortunately not successful. However, the results shown above were not convincing of achieving the same performance as provided in the paper for EfficientNet-B0. In the paper, all accuracies (Top-1 and Top-5) were higher than 78%. The results obtained in the first epochs of finetuning were not suggesting, that similar performances could have been obtained. However, this can be related to the fact that finetuning could not be finished, to an unideal checkpoint or to mistakes made while running the code, even if the tutorial was not altered and just run as given.

Next to that, it was also attempted to run different models, notably EfficientNetV2-S. Differently to the tutorial, the goal was to run it with a local copy of the EfficientNet model instead of downloading the model while running the code, in order to be able to adapt the files and implement the new dataset as well as the ablation study. However, even if all guidelines were followed, it was not possible to run the code when changing the model, the model directory or anything else. The errors were deep inside the EfficientNet model and therefore difficult to understand. However, from the errors shown, it was assumed that they are related to the fact, that the code was run on Google Colab and that not the original files were downloaded while running the code.

As numerous attempts to fix those issues were unsuccessful, the tensorflow implementation could unfortunately not be run. In consequence, an attempt was made to reproduce some results while setting up an implementation using PyTorch. This will be presented in the following section.

## Reproduction in PyTorch

### Description of PyTorch Implementation

### Training ImageNetTE

### Implementation of a New Dataset

### Ablation Study

### Analysing Comparability to Original Tensorflow Implementation

## Discussion on Reproducibility

## Conclusion

## Contributions

## Reference list

<a id="1">[1]</a> 
M. Tan and Q.V. Le (2021). 
EfficientNetV2: Smaller Models and Faster Training.
*Computing Research Repository, abs/2104.00298*.
https://doi.org/10.48550/arXiv.2104.00298.

<a id="2">[2]</a> 
Google. (2021). 
EfficientNetV2. 
Retrieved on 15th April 2023 from https://github.com/google/automl/tree/master/efficientnetv2.

<a id="3">[3]</a> 
Zalando Research. (2022). 
Fashion-MNIST. 
Retrieved on 15th April 2023 from https://github.com/zalandoresearch/fashion-mnist.
