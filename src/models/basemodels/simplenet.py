import os

import torch

import torch.nn as nn


import torchvision.transforms as T




class SimpleNet(nn.Module):
    def __init__(self,
                iamge_channels,
                output_dim 
                ) -> None:
        super(SimpleNet,self).__init__()
        self.image_channels = iamge_channels
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(in_channels=self.image_channels,
                               out_channels=32,
                               kernel_size=3,stride=2,
                               padding= 1
                               )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) 
        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size= (1,1))
        self.fc1 = nn.Linear(in_features=32,out_features=self.output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.avg_pool1(x)
        x = torch.flatten(x, 1)    
        x = self.fc1(x)
        return x


def simplenet(image_channels, output_dim):
    return SimpleNet(image_channels=image_channels, output_dim=output_dim)