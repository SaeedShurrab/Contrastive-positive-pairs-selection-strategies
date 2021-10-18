import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, 
                 clf: bool = True,
                ) -> None:
        super(SimpleNet,self).__init__()
        
        self.clf = clf
        
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,stride=2,
                               padding= 1
                               )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) 
        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size= (1,1))
        self.fc1 = nn.Linear(in_features=32,out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.avg_pool1(x)
        x = torch.flatten(x, 1)
        if self.clf:
            
            x = self.fc1(x)
        return x
