# EfficientNetV2 Reproducibility project

## Introduction

## Reproducibility Tensorflow Implementation

### Description of Original Tensorflow Implementation

### Implementation of a New Dataset

Several datasets are already implemented in the tensorflow implementation of the EfficientNetV2 model. The datasets implemented are: ImageNet, ImageNet21k, ImageNetTfds, Cifar10, Cifar100, Flowers, TFFlowers and Cars. In order to analyze if the implementation of EfficientNetV2 is adaptable to other datasets as well, an attempt was made to introduce a new dataset: FashionMNIST [[2]](#2). FashionMNIST is a dataset containing 10 different classes of Zalando product types, with in total 60000 testing and 10000 validation examples. They are all provided as grayscaled images with dimensions of 28x28 pixels.

In order to implement a new type of dataset, several changes needed to be made to the file *datasets.py*. First of all, a new class needed to be created for the new input of FashionMNIST. Similarly, to the other datasets, the new class was created by copying an older class and only changing the relevant entries in the dictionary. The new implementation was based on the class CIFAR10Input, which was the class used in the tutorial and thus (in theory) proven to be working. In the configuration dictionary of the new FashionMNIST class, the following elements were updated:

* The number of classes was set to 10.
* The dataset name was set to *fashion_mnist*, which is already implemented in tensorflow.
* The number of training images was set to 60000.
* The number of validation images was set to 10000 (minival, eval).

The implementation of the new class in the code can be seen below.

```ruby
class FashionMNISTInput(ImageNetInput):
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
class FashionMNISTFt(ImagenetFt):
  """Finetune fashionmnist configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='fashion_mnist'))
```

As the dataset of FashionMNIST is already implemented in tensorflow, those changes should be sufficient in order to implement this new dataset for EfficientNetV2. The old file *datasets.py* was thus then replaced by the new file containing the new implementations for FashionMNIST. 

### Possible Ablation Study

Next, an ablation study was supposed to be performed. As introduced in the article, an improvement in this new model was the progressive learning using an adaptive regularization. The image size was adpted in line 468 to 475 of the file *main.py*. To be very precise, the computations required were performed in line 472 to 474 and are shown as follows [[1]](#1).

```ruby
ratio = float(stage + 1) / float(total_stages)
max_steps = int(ratio * train_steps)
image_size = int(ibase + (input_image_size - ibase) * ratio)
```

The idea was to compare in the ablation study, how training would have performed when fully removing progressive learning with an adaptive regularization. Therefore, the code block shown above would need to be replaced with the one following, in order to have a constant image size. 

```ruby
image_size = input_image_size
```

Next to that, also the adaptive regularization would need to be removed. Regularization was found to also be performed in the file *main.py* from line 126 to 142. Again, some lines in the code were found to be most relevant for the regularization, namely line 129 to 131 defining the weight_decay as follows [[1]](#1).

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

## Reproduction in PyTorch

### Description of PyTorch Implementation

### Training ImageNetTE

### Implementation of a New Dataset

### Ablation Study

### Analysing Comparability to Original Tensorflow Implementation

## Conclusion

## Contributions

## Reference list

<a id="1">[1]</a> 
Google. (2021). 
EfficientNetV2. 
Retrieved on 15th April 2023 from https://github.com/google/automl/tree/master/efficientnetv2.

<a id="2">[2]</a> 
Zalando Research. (2022). 
Fashion-MNIST. 
Retrieved on 15th April 2023 from https://github.com/zalandoresearch/fashion-mnist.
