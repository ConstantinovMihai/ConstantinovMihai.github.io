# DRAFT: EfficientNetV2 Reproducibility project


## 1. Introduction

The paper “EfficientNetV2: Smaller Models and Faster Training“ [[1]](#1) was published in 2021 by M. Tan and Q.V. Le, researchers at Google, and introduces a new family of convolutional networks called EfficientNetV2. It builds on top of the original EfficientNetV1 architecture, which achieved state-of-the-art performance in several computer vision tasks. To develop these models, the authors introduced a combination of training-aware neural architecture search and scaling (NAS), as well as adaptive regularisation and a new building block called Fused-MBConv. They manage to achieve higher accuracy than the state-of-the-art models while training significantly faster and using 6.8 times fewer parameters.

As this paper introduced such a powerful architecture, the goal of this blog post is to briefly present the paper, as well as our attempt to reproduce several claims from the paper, as well as some ablation studies, to test the impact of the adaptive regularization and the new building block MBFused. Initially, it was tried to reproduce the results of the paper as well as to adapt the original code provided by the authors in TensorFlow. However, due to computational complexity and compatibility issues, it was not possible to obtain results with TensorFlow. Therefore, an alternative implementation of the code was created within PyTorch. Overall, the code and the results however proved to be hard to be reproduced.

The blog post is structured as follows: First, the most important achievements of the paper will be presented. Subsequently, the TensorFlow implementation is discussed and the changes to the provided code are presented. After explaining why it was unfortunately not possible to achieve results with the TensorFlow implementation, the implementation in PyTorch is presented and the results obtained are outlined. Finally, a final discussion of the reproducibility of the paper is given.
 

## 2. Theoretical Background and Summary of the Paper

The main contributions of the paper [[1]](#1) are the introduction of the new family EfficientNetV2, which by clever scaling and training aware NAS outperforms previous models in terms of training speed and memory requirements; a new method of progressive learning, which adjusts the regularization and the image size during training, which speeds up training and improves accuracy simultaneously. The researchers performed experiments to prove their new model beats the state of the art on a number of computer vision tasks.

As mentioned, EfficientNetV2 is built on top of the EfficientNet family, which was optimized for FLOPs and parameter efficiency, and constituted the state of the art due to their good trade-off on the accuracy, FLOPs, and parameter efficiency.

A bottleneck of the ImageNet dataset, a very popular computer vision task, is that the large image size makes training very slow. This problem was encountered by us, as described later in the blog post. Before EfficientNetV2, the solution employed was to downsize the training set images. However, the paper explores a new approach, by progressively adjusting during training the image size and the regularisation. 

The authors discuss the fact that depthwise convolutions are slow in early layers, but effective in later stages, which represents a bottleneck for EfficientNet, as it makes extensive use of such architectures. Even though they use fewer FLOPs, they can not fully utilize modern accelerators. In order to address this issue, Fused-MBConv is used instead of the MBConv (pictured in Figure 1) used in early layers in EfficientNetV2.  

| ![image](https://user-images.githubusercontent.com/97915789/232248614-4509fc17-09d4-47a9-946c-0582c256641b.png)| 
|:--:| 
| **Figure 1:** Structure of MBConv and Fused-MBConv [[1]](#1).|

EfficientNet scaled up all stages equally, which is suboptimal. Instead, EfficientNetV2 makes use of a non-uniform scaling strategy to gradually add more layers to later stages. Moreover, the maximum image size is restricted in order to avoid excessive memory consumption and slow training. NAS search is employed to select design choices such as the kernel sizes, number of layers, or the convolutional operation types (MBConv vs Fusd-MBConv). EfficientNetV2 architecture is presented in the table of Figure 2. Compared to EfficientNet, it makes extensive use of Fused-MBConv, prefers smaller expansion ratios for MBConv, and prefers smaller kernel size, but adds more layers to compensate for the reduction in the receptive field.

| **Table 1:** EfficientNetV2-S architecture – MBConv and FusedMBConv blocks are described in Figure 1 [[1]](#1). |
|:--:| 
| ![image](https://user-images.githubusercontent.com/97915789/232249289-1f523aed-bd8f-420f-8c9c-f2162d381ae5.png)| 

Progressive learning with adaptive regularization is achieved by training the network with smaller images and weak regularization in the early stages of the training, such that the network can learn simple regularisations easily. As training progresses, the image size as well as the regularisation increases in order to make learning more difficult. The pseudo-code of the progressive learning algorithm is presented in Figure 3.

| ![image](https://user-images.githubusercontent.com/97915789/232248170-7f2bde66-0958-4b14-9e31-9e2fad49a78e.png)|
|:--:| 
| **Figure 2:** [[1]](#1). |

## 3. Reproducibility TensorFlow Implementation

Having given a short summary of the paper, in this section the implementation of the paper in TensorFlow will be discussed. Furthermore, it is explained how new datasets can be implemented and how an ablation study could be done. The section ends with an overview of the problems encountered while attempting to run the TensorFlow model.

### 3.1. Description of Original TensorFlow Implementation

The original code [[2]](#2) is made public by the paper's authors and contains the entire implementation for both the EfficientNet and EfficientNetV2 models. It contains two main files corresponding to the two different TensorFlow versions. Unfortunately, the code is difficult to navigate and understand, as it is barely documented. Some parts, such as the adaptive regularisation, seem to be implemented in multiple parts of the code and it is not intuitive where the algorithms presented in the paper are implemented in the code. A large number of files containing utility functions, flags, as well as various tests might be of interest to a researcher willing to replicate the code. 

Three files are however of particular importance for this reproducibility project and will therefore be discussed in more detail. They are the following:

* *main.py*
* *hparams.py*
* *datasets.py*

First of all, the file *main.py* contains the main implementation of the training script. It imports the information and settings from all other files in the repository in order to train the selected EfficientNet model on the dataset. It builds the model, includes the loss function with the adaptive regularization, and runs both the training and evaluation. This file needs to be run when training the model. Furthermore, it contains flags with which before running for example the dataset configuration, the directory of checkpoints, or the chosen model can be defined.

Next, the file *hparams.py* contains all hyperparameters for the model architecture and for the training. They are related to the model, the training, the validation, the dataset used, and running everything. They can be changed in the *hparam_str* flag of the file *main.py* or immediately in the hyperparameters file itself. Furthermore, this file can simply be used to get an overview of all chosen hyperparameters.

Finally, the file *datasets.py* is of importance while training the model. In this file, the configurations for all available datasets need to be defined. They are used as input both for the training and for the validation. When defining the configurations, hyperparameters from the file *hparams.py* can possibly be redefined. Those datasets can then be called in the *dataset_cfg* flag of the file *main.py*. In the next subsection, it is discussed in detail how new dataset configurations can be defined.

### 3.2. Implementation of a New Dataset

Several datasets are already implemented in the TensorFlow implementation of the EfficientNetV2 model. The datasets implemented are ImageNet, ImageNet21k, ImageNetTfds, Cifar10, Cifar100, Flowers, TFFlowers, and Cars. In order to analyze if the implementation of EfficientNetV2 is adaptable to other datasets as well, an attempt was made to introduce a new dataset: FashionMNIST [[1]](#1). FashionMNIST is a dataset containing 10 different classes of Zalando product types, with in total of 60000 testing and 10000 validation examples. They are all provided as grayscaled images with dimensions of 28x28 pixels.

In order to implement a new type of dataset, several changes needed to be made to the file *datasets.py*. First of all, a new class needed to be created for the new input of FashionMNIST. Similarly, to the other datasets (the new implementation is inspired by the datasets already implemented [[1]](#1)), the new class was created by copying an older class and only changing the relevant entries in the dictionary. The new implementation was based on the class CIFAR10Input, which was the class used in the tutorial and thus (in theory) proven to be working. In the configuration dictionary of the new FashionMNIST class, the following elements were updated:

* The number of classes was set to 10.
* The dataset name was set to *fashion_mnist*, which is already implemented in tensorflow.
* The number of training images was set to 60000.
* The number of validation images was set to 10000 (minival, eval).

The implementation of the new class in the code can be seen below.

```tsql
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

```tsql
'fashion_mnist': FashionMNISTInput,
```

Finally, similarly to the other datasets a training configuration needed to be registered. This was again done by copying the training configuration *Cifar10Ft.cfg* of Cifar10 to the new class *FashionMNISTFt* for the configuration of FashionMNIST. The configuration data was overwritten with the data which was previously defined in the class of FashionMNIST. This was done as follows.

```tsql
@ds_register
class FashionMNISTFt(Cifar10Ft):
  """Finetune fashionmnist configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='fashion_mnist'))
```

As the dataset of FashionMNIST is already implemented in TensorFlow, those changes should be sufficient in order to implement this new dataset for EfficientNetV2. The old file *datasets.py* was thus then replaced by the new file containing the new implementations for FashionMNIST. 

### 3.3. Possible Ablation Study

Next, an ablation study was supposed to be performed. As introduced in the article, an improvement in this new model was progressive learning using adaptive regularization. The image size was adapted in lines 468 to 475 of the file *main.py*. To be very precise, the computations required were performed in lines 472 to 474 and are shown as follows [[2]](#2).

```tsql
ratio = float(stage + 1) / float(total_stages)
max_steps = int(ratio * train_steps)
image_size = int(ibase + (input_image_size - ibase) * ratio)
```

The idea was to compare in the ablation study, how training would have performed when fully removing progressive learning with an adaptive regularization. Therefore, the code block shown above would need to be replaced with the one following, in order to have a constant image size. 

```tsql
image_size = input_image_size
```

Next to that, also the adaptive regularization would need to be removed. Regularization was found to also be performed in the file *main.py* from lines 126 to 142. Again, some lines in the code were found to be most relevant for the regularization, namely lines 129 to 131 defining the weight_decay as follows [[2]](#2).

```tsql
weight_decay_inc = config.train.weight_decay_inc * (
    tf.cast(global_step, tf.float32) / tf.cast(train_steps, tf.float32))
weight_decay = (1 + weight_decay_inc) * config.train.weight_decay
```

Similar to the image size, a new version of those code lines would need to be implemented. The new line of code replacing the lines above and defining the weight decay in a different way in order to remove adaptive regularization would have been the following.

```tsql
weight_decay = 1
```

With those two changes, it would be possible to test how training would perform without using adaptive regularization for progressive learning. The difficulties faced while attempting to train the model, which prevented running the ablation study and the implementation of a new dataset, are described in the following subsection.

### 3.4. Problems With TensorFlow Implementation

As was described in the previous subsection, in this reproducibility project an attempt was made to reproduce the results presented in the paper on the model EfficientNetV2 [[1]](#1), as well as to try implementing a new dataset as well as performing an ablation study by removing adaptive regularization. However, unfortunately, several issues were faced while trying to run the code provided in the paper. They are summarized as follows.

In the repository provided (a link to the GitHub repository was given in the paper [[1]](#1), a tutorial was given on how to run the different types of the EfficientNet model with the files provided in the repository. In this tutorial, the model EfficientNet-B0 is finetuned. As a first step, it was therefore tried to run this tutorial. When strictly running what was given in the tutorial, training of the model was initialized and started running. Special attention was given to the Top-1 and Top-5 accuracy. In the first 56 of 781 training epochs, the Top-1 accuracy was observed to stay below 15% and the Top-5 accuracy varied between 45% and 55% but did not exceed 55%. It even decreased again for multiple epochs. The final line obtained can be seen below.

```tsql
56/781 [=>............................] - ETA: 149:30:16 - loss: 3.2239 - acc_top1: 0.1239 - acc_top5: 0.5413
```

Unfortunately, after running the code for nearly 1 day, it crashed. The computing power of a single computer was seen to not be sufficient to train the model in a reasonable timeframe. Attempts to run it in the Google Cloud were unfortunately not successful. However, the results shown above were not convincing of achieving the same performance as provided in the paper for EfficientNet-B0. In the paper, all accuracies (Top-1 and Top-5) were higher than 78%. The results obtained in the first epochs of finetuning were not suggesting, that similar performances could have been obtained. However, this can be related to the fact that finetuning could not be finished, to an unideal checkpoint, or to mistakes made while running the code, even if the tutorial was not altered and just run as given.

Next to that, it was also attempted to run different models, notably EfficientNetV2-S. Differently from the tutorial, the goal was to run it with a local copy of the EfficientNet model instead of downloading the model while running the code, in order to be able to adapt the files and implement the new dataset as well as the ablation study. However, even if all guidelines were followed, it was not possible to run the code when changing the model, the model directory, or anything else. The errors were deep inside the EfficientNet model and therefore difficult to understand. However, from the errors shown, it was assumed that they were related to the fact, that the code was run on Google Colab and that not the original files were downloaded while running the code.

As numerous attempts to fix those issues were unsuccessful, the TensorFlow implementation could unfortunately not be run. In consequence, an attempt was made to reproduce some results while setting up an implementation using PyTorch. This will be presented in the following section.

## 4. Reproduction in PyTorch

Due to the impact the EfficientNetV2 family had on the computer vision community, the torchvision package has an implementation in the PyTorch library of the model. We tried using it for the ablation studies and for the new data set, by creating a training loop. Our efforts are documented in this section. 

### 4.1. Description of PyTorch Implementation

In this section, a short description of the PyTorch training loop will be given. An explanation of how to load the datasets for training will follow in the subsequent sections. In the first step, loading the model from Torchvision's library is done as follows:
```tsql
from torchvision.models import efficientnet_v2_s

# Define the EfficientNet_V2_S model
model = efficientnet_v2_s()
```
To prepare the training loop, a loss criterion, based on Cross Entropy (as FashionMNIST is a classification task), learning rate, optimizer (Adam is chosen for this task, due to its popularity and good properties), and a scheduler for the weight decay are defined:

```tsql
# Define the loss function
criterion = nn.CrossEntropyLoss()
lr = 0.005
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

The training loop is built by, firstly, choosing a number of epochs and iterating for the number of epochs:
```tsql
# Train the model
num_epochs = 100

for epoch in range(num_epochs):
```

For each epoch, the training loss and training accuracy are defined:
```tsql
    train_loss = 0.0
    train_correct = 0.0
```

Further, during one epoch we iterate among all the batches, performing a full pass(forward + backward pass). This is done by first initializing the gradients to zero:
```tsql
       for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
```

Afterward, the forward pass is performed:
```tsql
        output = model(data)
```
The backward pass is performed by firstly computing the loss, and then running the backward pass:
```tsql
        loss = criterion(output, target)
        loss.backward()
```

Adam uses the gradients computed during the backpropagation step to update the weights:
```tsql
      optimizer.step()
```

The statistics, i.e. the accuracy and loss, for the training set are computed using:
```tsql
     # Update statistics
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == target).sum().item()
```

For each epoch, the test set is used to generate statistics regarding the performance of the training. Similar to the train statistics case, first calling torch.eval() method, and declaring the test accuracy and loss:
```tsql
     # Calculate statistics for the validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0.0
```

Crucially for this step, the backpropagation should not be called, and therefore the forward pass is performed by explicitly telling the program not to compute the gradients:

```tsql
     # Calculate statistics for the validation set
    with torch.no_grad():
```

Similarly, with the training case, data is loaded by batches, and the forward pass is performed:

```tsql
      for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
```

Loss is computed and the statistics are updated:
```tsql
      loss = criterion(output, target)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_correct += (predicted == target).sum().item()
```

Finally, the train and test statistics are printed for the user to check the progress of the training:
```tsql
    # Print the training and validation statistics for the epoch
    train_loss /= len(train_loader.dataset)
    train_acc = 100.0 * train_correct / len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)
    val_acc = 100.0 * val_correct / len(test_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

### 4.2. Producing Results by Training on ImageNetTE

In the first step, the model implemented in PyTorch was trained on ImageNetTE. This is the most similar dataset to ImageNet, which was the dataset used in the original paper [[1]](#1)). More precisely, it is a subset containing 10 classes of the original ImageNet dataset [[4]](#4). As it is not integrated into the PyTorch datasets, the training and validation images were downloaded online [[4]](#4) and were inserted in the same folder as the Python file for training the EfficientNetV2-s model.

First, in order to be able to use images from a downloaded folder for training the model, the *ImageFolder* training function needed to be imported from torchvision. This was done as follows.

```tsql
from torchvision.datasets import ImageFolder
```

Next, a transform needed to be defined for this dataset. Inspiration for the transform was taken from [[5]](#5) which uses the transforms *CenterCrop*, *ToTensor*, and *Normalize* to transform the original ImageNetTE images. It can be seen below how the transformation was defined.

```tsql
transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Finally, from the downloaded images, the training and testing dataset needed to be created. This was done as follows, by searching for the images in the download folder and by applying the transform which was defined before.

```tsql
train_dataset = ImageFolder(root='./data/imagenette2-160/train', transform=transform)
test_dataset = ImageFolder(root='./data/imagenette2-160/val', transform=transform)
```

Having implemented the ImageNetTE dataset, training was now performed with different datasets and batch sizes. Unfortunately, it was not possible to run it on the entire dataset, as this took more than a night for only one epoch. Therefore, it was decided to train on several subset sizes of the full dataset, in order to understand the influence of the size of the dataset and to draw conclusions from that.

As can be seen in the following figures, different subsets of the dataset ImageNetTE were trained and validated using the model EfficientNetV2-s. The sizes chosen were 64, 128, 256, and 512 images always respectively for training and validation. They were always divided into batches of the size of 32 images. For this example, it was chosen to train the model on 10 epochs due to limitations in time. The results obtained for the training and validation accuracy can be observed in the following Figures 4 and 5 respectively.

| <img width="100%" alt="INTETrainingAccuracy" src="https://user-images.githubusercontent.com/74194871/232327384-49e41762-864e-48a5-b502-55830369ebed.png">| <img width="100%" alt="INTEValidationAccuracy" src="https://user-images.githubusercontent.com/74194871/232327415-53dffcde-44d4-4f4c-ab14-aa7c0d5154a4.png">|
|:--:|:--:| 
| **Figure 4:** Training accuracy per epoch on different training dataset sizes. | **Figure 5:** Validation accuracy per epoch on different training dataset sizes.|

As can be seen from the figures above, the training accuracy increases when increasing the number of images in the datasets. For datasets with a small number of images (up to 512 images), it always reaches a training accuracy of 100% after some epochs, but that would most probably not be the case when training on the full training dataset. In fact, for larger training dataset sizes the accuracy does not reach 100% anymore. Furthermore, it can be seen that the validation accuracy stays more constant from epoch to epoch, especially for the dataset size of 512, where it has a constant value of 75.59%. However, it can be seen that with increasing validation dataset sizes, accuracy decreases drastically. This could however be avoided probably by using more images for training than for validation.

Furthermore, it is observed that the training loss as well as the validation loss decreases with increasing dataset sizes. This can be seen in the following Figures 6 and 7. Similarly to the training accuracies, for the training loss in most of the cases, it decreases to 0 after a few epochs. For the validation loss, the trend is less clear. For the 64 images dataset size it can clearly be seen that it slowly decreases to 0. However, for the other dataset sizes, it always first increases to values higher than 0.1 before then staying constant or decreasing to 0 again.

| <img width="100%" alt="INTETrainingLoss" src="https://user-images.githubusercontent.com/74194871/232327436-cb417b4f-dc2a-4b71-8298-4c9e6edeaf4f.png">| <img width="100%" alt="INTEValidationLoss" src="https://user-images.githubusercontent.com/74194871/232327457-0e754466-7521-4fd1-af59-b5e24a17d99a.png">|
|:--:|:--:| 
| **Figure 6:** Training loss per epoch on different training dataset sizes. | **Figure 7:** Validation loss per epoch on different training dataset sizes.|

The results (especially the increasing loss for higher dataset sizes) suggest that overfitting occurs while training the datasets. This could arise from the fact that a pre-trained model on ImageNet was used in order to train ImageNetTE, which was explained to be a subset of ImageNet [[4]](#4)). Thus, very high accuracies can be achieved. In order to verify if this would not be the case for other datasets and if training results are as expected for other datasets, the next training was implemented on FashionMNIST. This will be described in the following subsection.

### 4.3. Implementation of a New Dataset

As a new dataset, the FashionMNIST was chosen as it is a popular and rather lightweight model. Firstly, the datasets are obtained from torchvision's dataset and loaded into the program. For instance, FashionMNIST is loaded as follows (the implementation is inspired from [[7]](#7)):

```tsql
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

For the sake of visualization, only two classes from FashionMNIST are plotted. Firstly, selecting the indices for the second and third class from the targets of the train and test data sets:

```tsql
idx = (train_dataset.targets == 1) | (train_dataset.targets == 2)
```

Now, using only the train and test data that has the desired targets using:
```tsql
train_dataset.data = train_dataset.data[idx]
train_dataset.targets = train_dataset.targets[idx]

test_dataset.data = test_dataset.data[idx]
test_dataset.targets = test_dataset.targets[idx]
```

Selecting a subset of the entire dataset is done using:
```tsql
train_dataset = Subset(train_dataset, range(16))
test_dataset = Subset(train_dataset, range(16))
```

Data is loaded using a DataLoader, by setting a batch size and the shuffle boolean:
```tsql
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
```

In order to decode the labels to actual clothing pieces, the following dictionary is used:

```tsql
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
```

Plotting the FashionMNISt images, to obtain the plots from Figure 3:

```tsql
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
| **Figure 3:** FashionMNIST instances. |

To make images from different datasets compatible with the EfficientNetV2 class, a transform has to be applied. For instance, to resize the image and increase the number of channels the following snippet might be used:

```tsql
# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(224), # resize the image to 224x224
    transforms.Grayscale(num_output_channels=3), # convert the image to RGB format
    transforms.ToTensor(), # convert the image to a PyTorch tensor
])
```

The transform might be used directly by the DataLoader. Several experiments were run, varying train set sizes, regularization, weights used, etc. When the model does not use pre-trained weights, it will perform poorly as, due to its huge size, it needs plenty of training in order to even learn how to overfit the training set. The experiments showed that, after 100 epochs, the model had accuracy in the same order of magnitude as random guessing. Using pre-trained weights generates very good results, even for a small number of epochs and a relatively low train set size, indicating that the original model's parameter performs well on the new task. As can be seen in Figure 9, early stopping as a regularization can render good results while saving on computation. Even though the training loss decreases in Figure 10, it does not generalize to the training data in Figure 11. It is worth mentioning that running the model with pre-trained weights and using weight decay produces unsatisfactory results, similar to random guessing. Apart from weight decay, other regularization techniques might be considered in the future, as the model seems to overfit on the training set, in Figure 8. By using smaller train set sizes, such as 512 or 256 images, the results obtained are very similar, at a much lower computational price.

| <img width="100%" alt="FMTrainingAccuracy" src="https://user-images.githubusercontent.com/74194871/232315401-cff9ae94-9eb0-4dda-82e1-636f3aa1908d.png">| <img width="100%" alt="FMValidationAccuracy" src="https://user-images.githubusercontent.com/74194871/232315420-109cb863-b23d-4aa5-ad38-f8407140b1db.png">|
|:--:|:--:| 
| **Figure 8:** Training accuracy per epoch on FashionMNIST. | **Figure 9:** Validation accuracy per epoch on FashionMNIST.|

| <img width="100%" alt="FMTrainingLoss" src="https://user-images.githubusercontent.com/74194871/232315447-86736bb8-b1a8-4ec3-9210-22f3e9ff756d.png">| <img width="100%" alt="FMValidationLoss" src="https://user-images.githubusercontent.com/74194871/232315463-0cd9f771-4424-43e6-91f4-e507d25a9856.png">|
|:--:|:--:| 
| **Figure 10:** Training loss per epoch on FashionMNIST. | **Figure 11:** Validation loss per epoch on FashionMNIST.|

### 4.4. Ablation Study

One of the features of EfficientNetV2 architecture especially highlighted by authors is the use of a Fused-MBConv module in the earlier layers, rather than MBConv. We perform an ablation study to attempt to verify whether this choice of types of modules with convolutional layers is the source of the increase in performance. In the ablation study the layers previously using Fused-MBConv are altered to use MBConv. The changes are made within the model implementation in PyTorch, by altering the lines defining the configuration of the network layers, as seen in the following snippet:

```tsql
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            # FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            # FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            # FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(1, 3, 1, 24, 24, 2),
            MBConvConfig(4, 3, 2, 24, 48, 4),
            MBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
```

The ablation study is performed for the EfficientNetV2-S model due to its smaller size and thus shorter training times. The sizes of the convolutions and the number of layers were not changed, as the primary aim of the ablation study is to verify the effectiveness of Fused-MBConv vs MBConv module architecture. The experiment is performed on the FashionMNIST dataset and using the same part of the whole set as previously. The results of the original architecture vs the modified one can be seen in the figures below:

| <img width="100%" alt="FMTrainingAccuracy" src="https://user-images.githubusercontent.com/79273017/232326916-1e1a56cc-1d85-443c-83b3-7e5377d524ee.png">| <img width="100%" alt="FMValidationAccuracy" src="https://user-images.githubusercontent.com/79273017/232326938-ffedd8a2-2fa1-4639-be01-33e1ec146d8d.png">|
|:--:|:--:| 
| **Figure 8:** Training loss per epoch on modified network. | **Figure 9:** Training accuracy per epoch on modified network.|

| <img width="100%" alt="FMTrainingLoss" src="https://user-images.githubusercontent.com/79273017/232326961-18b5bb82-8f52-4c97-837a-0c6d8694b623.png">| <img width="100%" alt="FMValidationLoss" src="https://user-images.githubusercontent.com/79273017/232327007-e30aaaaa-3554-457e-8b55-82ee4b8b72a8.png">|
|:--:|:--:| 
| **Figure 10:** Validation loss per epoch on the modified network. | **Figure 11:** Validation accuracy per epoch on the modified network.|

The effect of the changes can be seen mostly in the accuracy of the validation data. The modified architecture using MBConv reachers has lower accuracy than the original one proposed by the authors. The performance on the training set is very similar so the effect of the change on the training performance is not deemed significant. Similarly, the losses exhibit similar numbers. The lower accuracy on the validation dataset can indeed be attributed to the choice of the convolution method. This aligns with the author's findings on the effectiveness of using the Fused-MBConv over MBConv in the early network layers. One feature which can be noted is that the accuracy of the original network tends to fluctuate a bit more over the epochs while the modified architecture exhibits a more stable trend. However, it is not fully clear whether that is a contribution to this ablation study. 

To conclude this experiment it was found that the proposed architecture of convolution modules indeed performs better. As all other parameters of the network architecture were kept constant it can be concluded that using Fused-MBConv instead of MBConv is a source of performance gain. However, each convolution module features plenty of parameters that were not changed when altering the module type. It is possible that if parameters such as kernel size or stride would be changed to values better suited for MBConv in the early layers the network could have achieved better performance. Such analysis is suggested for future studies.

### 4.5. Analysing Comparability to Original Tensorflow Implementation

The experiments in Pytroch use a model implemented as a part of the pytorch vision package. It is possible that there are small differences exhibited in the pytorch implementation, possibly related to the clarity or reusability of the model. The TensorFlow implementation is the original proposal of this architecture therefore the results obtained with the original code could be closer to the ones obtained by the authors. However, as EfficientNetV2 is an architecture that has found many uses in additional studies after it was conceived for the first time. Therefore it can be assumed that performing experiments with the model implemented in Pytroch is able to provide meaningful insights into the performance and characteristics of the EfficientNetV2 architecture.

## 5. Discussion on Reproducibility

After putting considerable time and effort into attempting to reproduce the EfficientNetV2 paper, one must consider what are the causes of the numerous difficulties encountered during the project. While our inexperience no doubt played a major role in it, we consider that the paper is on the higher end in terms of difficulty of reproducibility even for the more seasoned researcher.

Firstly, the computational resources needed to reproduce the author's results, even when finetuning the model, are out of reach for most people, especially students. Secondly, apart from the tutorial provided in the repository, the code is not documented and one has difficulties navigating and understanding it. Moreover, the original code is written in the TensorFlow framework, parts of which are deprecated, and we encountered numerous compatibility issues between different versions of different libraries. The torch-vision model of EfficientNet circumvents this problem, but no proper documentation is provided for people new in the field to understand how to use it.

In conclusion, this project made us realize that merely sharing the code is not enough to ensure the results are reproducible. Reproducibility is a crucial step to ensure the real value of the results presented in the paper, especially in light of [6]). However, as EfficientNetV2 is a famous family model, probably the latter point is not the case in our situation. Nonetheless, especially for such a seminal paper, we consider it should be more straightforward to at least be able to verify the paper's results.

## 6. Conclusion

Attempting to run and modify the original TensorFlow implementation as well as performing experiments with our own Pytorch implementation has given us an insight into the proposed architecture and studies proposed by the authors. While experiments on the original TensorFlow implementation were mostly unsuccessful it has been a valuable experience on the reproducibility of scientific papers, even when the full code has been released. Our implementation based mostly on the PyTorch vision package has allowed us to work with the proposed architecture and evaluate it on new datasets as well as perform an ablation study related to the architecture of the modules. The results obtained on the Pytorch model did not show any features which would dispute the authors' findings.

## Contributions

|Part of the work|Mihai Constantinov|Christoph Pabsch|Igor Pszczólkowski|
|:--:|:--:|:--:|:--:|
|Attempts to Run TensorFlow|X|X|X|
|Implementation FashionMNIST TensorFlow|X|X||
|Explanations Problems TensorFlow||X||
|Setting up PyTorch Training Loop|X|||
|PyTorch ImageNetTE Implementation & Analysis||X||
|PyTorch FashionMNIST Implementation & Analysis|X|X||
|PyTorch Ablation Study|||X|
|Overall Reproducibility Analysis|X|X|X|

## Reference list

<a id="1">[1]</a> 
M. Tan and Q.V. Le. (2021). 
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

<a id="4">[4]</a> 
fast.ai. (2022). 
Imagenette. 
Retrieved on 16th April 2023 from https://github.com/fastai/imagenette.

<a id="5">[5]</a> 
Aman Amora. (2021). 
Distributed Training in PyTorch on ImageNette. 
Retrieved on 16th April 2023 from https://github.com/amaarora/imagenette-ddp/blob/master/src/config.py.

<a id="6">[6]</a> 
Z. Lipton & J. Steindhardt. (2018). 
Troubling trends in machine learning scholarship. 
*arXiv preprint arXiv:1807.03341*.
https://arxiv.org/abs/1807.03341

<a id="7">[7]</a> 
pytorch.org. 
torchvision. 
Retrieved on 16th April 2023 from https://pytorch.org/vision/stable/index.html.

