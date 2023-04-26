"""This file contains the implementation in PyTorch for ImageNetTE.
ImageNetTE needs to be added in a data folder in the same folder.
ImageNetTE can be downloaded from https://github.com/fastai/imagenette (160px download)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s
from torch.optim import lr_scheduler

# Define the transformation pipeline
# (Inspired from: https://github.com/amaarora/imagenette-ddp/blob/master/src/config.py)
transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Create the FashionMNIST datasets with the transformation applied
train_dataset = ImageFolder(root='./data/imagenette2-160/train', transform=transform)
test_dataset = ImageFolder(root='./data/imagenette2-160/val', transform=transform)

N = 1024
bs = 32
train_dataset = Subset(train_dataset, range(N))
test_dataset = Subset(test_dataset, range(N))


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

# Define the EfficientNet_V2_S model
model = efficientnet_v2_s(pretrained=True)

# Define the loss function
criterion = nn.CrossEntropyLoss()
lr = 0.001
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
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

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')