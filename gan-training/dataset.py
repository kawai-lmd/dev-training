import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from model import *

epochs = 500
batch_size = 256

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root="./data",
                                            train=True, download=True,
                                            transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, pin_memory=True)

datas = torchvision.datasets.CIFAR10(root="./data",
                                            train=True, download=True,
                                            transform=None)

label_list = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

img_num = 40
print(label_list[datas[img_num][1]])
plt.imshow(datas[img_num][0])
plt.show()
