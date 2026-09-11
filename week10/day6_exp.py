import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# define transforms
transforms = transforms.Compose({
    transforms.ToTensor(),

    transforms.Normalize((0.5,),(0.5,))
})

# load datasets
train_dataset = datasets.MNIST(root = "./data",train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root="./data", train= False, transform=transforms, download=True)

# create data loader 
train_loder = DataLoader(train_dataset, batch_size=32, shuffle=True)
