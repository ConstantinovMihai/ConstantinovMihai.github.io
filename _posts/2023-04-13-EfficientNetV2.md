# EfficientNetV2 Reproducibility project


## Introduction

The paper “EfficientNetV2: Smaller Models and Faster Training“ [[1]](#1) was published in 2021 by M. Tan and Q.V. Le, researchers at Google, and introduces a new family of convolutional networks called EfficientNetV2. It builds on top of the original EfficientNetV1 architecture, which achieved state-of-the-art performance in several computer vision tasks. To develop these models, the authors introduced a combination of training-aware neural architecture search and scaling (NAS), as well as adaptive regularisation and a new building block called Fused-MBConv. They manage to achieve higher accuracy than the state-of-the-art models while training significantly faster and using 6.8 times fewer parameters.

As this paper introduced such a powerful architecture, the goal of this blog post is to briefly present the paper, as well as our attempt to reproduce several claims from the paper, as well as some ablation studies, to test the impact of the adaptive regularization and the new building block MBFused. Initially, it was tried to reproduce the results of the paper as well as to adapt the original code provided by the authors in TensorFlow. However, due to computational complexity and compatibility issues, it was not possible to obtain results with TensorFlow. Therefore, an alternative implementation of the code was created with in PyTorch. Overall, the code and the results however proved to be hard to being reproduced.

The blog post is structured as follows: First, the most important achievements of the paper will be presented. Subsequently, the TensorFlow implementation is discussed and the changes to the provided code are presented. After explaining why it was unfortunately not possible to achieve results with the TensorFlow implementation, the implementation in PyTorch is presented and the results obtained are outlined. Finally, a final discussion of the reproducibility of the paper is given.
 

## Theoretical Background and Summary of the Paper

The main contributions of the paper [[1]](#1) are the introduction of the new family EfficientNetV2, which by clever scaling and training aware NAS outperforms previous models in terms of training speed and memory requirements; a new method of progressive learning, which adjusts the regularization and the image size during training, which speeds up training and improves accuracy simultaneously. The researchers performed experiments to prove their new model beats the state of the art on a number of computer vision tasks.

As mentioned, EfficientNetV2 is built on top of the EfficientNet family, which was optimized for FLOPs and parameter efficiency, and constituted the state of the art due to their good trade-off on the accuracy, FLOPs, and parameter efficiency.

A bottleneck of the ImageNet dataset, a very popular computer vision task, is that the large image size makes training very slow. This problem was encountered by us, as described later in the blog post. Before EfficientNetV2, the solution employed was to downsize the training set images. However, the paper explores a new approach, by progressively adjusting during training the image size and the regularisation. 

The authors discuss the fact that depthwise convolutions are slow in early layers, but effective in later stages, which represents a bottleneck for EfficientNet, as it makes extensive use of such architectures. Even though they use fewer FLOPs, they can not fully utilize modern accelerators. In order to address this issue, Fused-MBConv is used instead of the MBConv (pictured in Figure 1) used in early layers in EfficientNetV2.  

| ![image](https://user-images.githubusercontent.com/97915789/232248614-4509fc17-09d4-47a9-946c-0582c256641b.png)| 
|:--:| 
| **Figure 1: Structure of MBConv and Fused-MBConv.** [[1]](#1)|

EfficientNet scaled up all stages equally, which is suboptimal. Instead, EfficientNetV2 makes use of a non-uniform scaling strategy to gradually add more layers to later stages. Moreover, the maximum image size is restricted in order to avoid excessive memory consumption and slow training. NAS search is employed to select design choices such as the kernel sizes, number of layers, or the convolutional operation types (MBConv vs Fusd-MBConv). EfficientNetV2 architecture is presented in the table of Figure 2. Compared to EfficientNet, it makes extensive use of Fused-MBConv, prefers smaller expansion ratios for MBConv, and prefers smaller kernel size, but adds more layers to compensate for the reduction in the receptive field.

| ![image](https://user-images.githubusercontent.com/97915789/232249289-1f523aed-bd8f-420f-8c9c-f2162d381ae5.png)| 
|:--:| 
| **Figure 2: EfficientNetV2-S architecture – MBConv and FusedMBConv blocks are described in Figure 1.** [[1]](#1) |

Progressive learning with adaptive regularization is achieved by training the network with smaller images and weak regularization in the early stages of the training, such that the network can learn simple regularisations easily. As training progresses, the image size as well as the regularisation increases in order to make learning more difficult. The pseudo-code of the progressive learning algorithm is presented in Figure 3.

| ![image](https://user-images.githubusercontent.com/97915789/232248170-7f2bde66-0958-4b14-9e31-9e2fad49a78e.png)|
|:--:| 
| **Figure 3.** [[1]](#1) |

## Reproducibility TensorFlow Implementation

Having given a short summary of the paper, in this section the implementation of the paper in TensorFlow will be discussed. Furthermore, it is explained how new datasets can be implemented and how an ablation study could be done. The section ends with an overview of the problems encountered while attempting to run the TensorFlow model.

### Description of Original TensorFlow Implementation

The original code [[2]](#2) is made public by the paper's authors, and contain the entire implementation for both the EfficientNet and EfficientNetV2 models. It contains two main files corresponding to the two different tensorflow versions. Unfortunately, the code is difficult to navigate and understand, as it is barely documented. Some parts, such as the adaptive regularisation, seem to be implemented in multiple parts of the code and it is not intuititve where the algorithms presented in the paper are implemented in the code. A large number of files containing utility functions, flags, as well as various tests might be of interest for a researcher willing to replicate the code. 

Three files are however of particular importance for this reproducibility project and will therefore be discussed in more detail. They are the following:

* *main.py*
* *hparams.py*
* *datasets.py*

First of all, the file *main.py* contains the main implementation of the training script. It imports the information and settings from all other files in the repository in order to train the selected EfficientNet model on the dataset. It builds the model, includes the loss function with the adaptive regularization and runs both the training and evaluation. This file needs to be run when training the model. Furthermore, it contains flags with which before running for example the dataset configuration, the directory of checkpoints or the chosen model can be defined.

Next, the file *hparams.py* contains all hyperparameters for the model architecture and for the training. They are related to the model, the training, the validation, the dataset used and running everything. They can be changed in the *hparam_str* flag of the file *main.py* or immediately in the hyperparameters file itself. Furthermore, this file can simply be used to get an overview of all chosen hyperparameters.

Finally, the file *datasets.py* is of importance while training the model. In this file, the configurations for all availabe datasets need to be defined. They are used as an input both for the training and for the validation. When defining the configurations, hyperparameters from the file *hparams.py* can possibly be redefined. Those datasets can then be called in the *dataset_cfg* flag of the file *main.py*. In the next subsection, it is discussed in detail how new dataset configurations can be defined.

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

### Problems With TensorFlow Implementation

As was described in the previous subsection, in this reproducibility project an attempt was made to reproduce the results presented in the paper on the model EfficientNetV2 [[1]](#1), as well as to try implementing a new dataset as well as performing an ablation study by removing adaptive regularization. However, unfortunately, several issues were faced while trying to run the code provided in the paper. They are summarized as follows.

In the repository provided (a link to the GitHub repositroy was given in the paper [[1]](#1)), a tutorial was given on how to run the different types of the EfficientNet model with the files provided in the repository. In this tutorial, the model EfficientNet-B0 is finetuned. In a first step, it was therefore tried to run this tutorial. When strictly running what was given in the tutorial, training of the model was initialized and started running. Special attention was given to the Top-1 and Top-5 accuracy. In the first 56 of 781 training epochs, the Top-1 accuracy was observed to stay below 10% and the Top-5 accuracy varied between 40% and 50%, but did not exceed the 50%. It even decreased again for multiple of the epochs. 

Unfortunately, after running the code for nearly 1 day, it crashed. The computing power of a single computer were seen to not be sufficient to train the model in a reasonable timeframe. Attempts to run it in the Google Cloud were unfortunately not successful. However, the results shown above were not convincing of achieving the same performance as provided in the paper for EfficientNet-B0. In the paper, all accuracies (Top-1 and Top-5) were higher than 78%. The results obtained in the first epochs of finetuning were not suggesting, that similar performances could have been obtained. However, this can be related to the fact that finetuning could not be finished, to an unideal checkpoint or to mistakes made while running the code, even if the tutorial was not altered and just run as given.

Next to that, it was also attempted to run different models, notably EfficientNetV2-S. Differently to the tutorial, the goal was to run it with a local copy of the EfficientNet model instead of downloading the model while running the code, in order to be able to adapt the files and implement the new dataset as well as the ablation study. However, even if all guidelines were followed, it was not possible to run the code when changing the model, the model directory or anything else. The errors were deep inside the EfficientNet model and therefore difficult to understand. However, from the errors shown, it was assumed that they are related to the fact, that the code was run on Google Colab and that not the original files were downloaded while running the code.

As numerous attempts to fix those issues were unsuccessful, the tensorflow implementation could unfortunately not be run. In consequence, an attempt was made to reproduce some results while setting up an implementation using PyTorch. This will be presented in the following section.

## Reproduction in PyTorch

Due to the impact EfficientNetv2 family had in the computer vision community, the torchvision package has an implementation in Pytorch library of the model. We tried using it for the ablation studies and for the new data set, by creating a training loop. Our efforts are documented in this section. 

### Description of PyTorch Implementation

Firstly, the datasets are obtaied from torchvision's dataset and loaded in the program. For instance, FashionMNIST is loaded as follows:

```ruby
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

For the sake of visualisation, the following snippet loads FashionMNIST data set and plots several instances from the second and third classes, as can be seen in Figure 4:

```ruby

idx = (train_dataset.targets == 1) | (train_dataset.targets == 2)
train_dataset.data = train_dataset.data[idx]
train_dataset.targets = train_dataset.targets[idx]


idx = (test_dataset.targets == 1) | (test_dataset.targets == 2)
test_dataset.data = test_dataset.data[idx]
test_dataset.targets = test_dataset.targets[idx]


train_dataset = Subset(train_dataset, range(16))
test_dataset = Subset(train_dataset, range(16))


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(9, 9))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = i
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```


| ![image](https://user-images.githubusercontent.com/97915789/232250211-f7e65d08-a003-413d-b38b-680e8bf6c8e0.png)|
|:--:| 
| **Figure 4. FashionMNIST instances.** |

To make images from different datasets compatible wih the EfficientNetV2 class, a trasform has to be applied. For instance, to resize the image and increase the number of channels the following snippet might be used:

```ruby
# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(224), # resize the image to 224x224
    transforms.Grayscale(num_output_channels=3), # convert the image to RGB format
    transforms.ToTensor(), # convert the image to a PyTorch tensor
])
```

The tranform might be used directly by the DataLoader.

Loading the model from torchvision's library is done as follows:
```ruby
from torchvision.models import efficientnet_v2_s

# Define the EfficientNet_V2_S model
model = efficientnet_v2_s()
```
To prepare the training loop, a loss criterion, based on Cross Entropy (as FashionMNIST is a classification task), learning rate, optimizer (Adam is chosen for this task, due to its popularity and good properties), and a scheduler for the weight decay are defined:

```ruby
# Define the loss function
criterion = nn.CrossEntropyLoss()
lr = 0.005
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

Finally, the training loop is defined by choosing the number of epochs, iterating among the batches, initialising the gradients to zero, performing the forward pass, and then the backward pass and updating the weigths based on the Adam optimizer. The training loss is computed. Moreover, in parallel and without performing the backward propagation step, the forward pass on validation data is used to compute the vaidation loss and accuracy.

```ruby
# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
  
        optimizer.step()
       
        # Update statistics
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == target).sum().item()

    # Calculate statistics for the validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_correct += (predicted == target).sum().item()

    # Print the training and validation statistics for the epoch
    train_loss /= len(train_loader.dataset)
    train_acc = 100.0 * train_correct / len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)
    val_acc = 100.0 * val_correct / len(test_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

### Training ImageNetTE

### Implementation of a New Dataset

### Ablation Study

### Analysing Comparability to Original Tensorflow Implementation

## Discussion on Reproducibility

## Conclusion

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
