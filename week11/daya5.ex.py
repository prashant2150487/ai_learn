import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
# define transfromation
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.50.5))

])

# load cifar10 dataset 
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,transform=transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms)


# Create data Loaders

