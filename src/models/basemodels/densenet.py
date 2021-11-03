
from typing import List
import torch

import torch.nn as nn
from torch import Tensor


class Bottleneck(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 growth_rate: int) -> None:
        super(Bottleneck,self).__init__()
        
        inner_channels = 4 * growth_rate
        
        
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace= True),
                                        nn.Conv2d(in_channels= in_channels, out_channels= inner_channels,
                                                  kernel_size= 1, bias= False),
                                        nn.BatchNorm2d(inner_channels),
                                        nn.ReLU(inplace= True),
                                        nn.Conv2d(in_channels= inner_channels, out_channels= growth_rate,
                                                  kernel_size= 3, padding= 1, bias= False))
        
    def forward(self,
                x: Tensor
                ) -> Tensor:
        return self._concat(x)
    
    def _concat(self,
                x: Tensor
                ) -> Tensor:
        return torch.cat([x, self.bottleneck(x)],1)


class TransitionBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels
                ) -> None:
        super(TransitionBlock, self).__init__()
        
        self.transition = nn.Sequential(nn.BatchNorm2d(in_channels), 
                              nn.ReLU(inplace= True),
                              nn.Conv2d(in_channels= in_channels, out_channels= out_channels,
                                                  kernel_size= 1, stride= 1,bias= False),
                              nn.AvgPool2d(kernel_size= 3, stride= 2, padding=0))
    def forward(self,
                x: Tensor
               ) -> Tensor:
        return self.transition(x)



class DenseBlock(nn.Module):
    def __init__(self,
                 inner_channels: int, 
                 n_layers: int,
                 growth_rate: int
                 ) -> None:
        super(DenseBlock,self).__init__()
        
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.inner_channels = inner_channels 
        
        self.denseblock = self._make_layer(self.n_layers,self.growth_rate)
        
    def forward(self,
                x: Tensor
               ) -> Tensor:
        return self.denseblock(x)
    
    
    def _make_layer(self, 
                    n_layers: int,
                    growth_rate: int
                   ) -> nn.Module:
        
        layers = []

        
        for i in range(self.n_layers):
            layers.append(Bottleneck(in_channels= self.inner_channels, growth_rate = self.growth_rate))
            self.inner_channels += self.growth_rate
        
        
        return nn.Sequential(*layers)



class DenseNet(nn.Module):
    def __init__(self, 
                 config: List,
                 growth_rate: int = 32,
                 image_channels: int = 3,
                 compression: float = 0.5, 
                 output_dim: int = 1000
                 ) -> None:
        super(DenseNet, self).__init__()
        
        n_layers = config
        
        self.image_channels = image_channels
        self.output_dim = output_dim
        self.growth_rate = growth_rate
        self.compression = compression
        
        
        self.inner_channels = 2 * self.growth_rate
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels= self.image_channels, 
                                             out_channels= self.inner_channels,kernel_size= 7, stride=2, 
                                             padding=3, bias= False),
                                   nn.BatchNorm2d(self.inner_channels),
                                   nn.ReLU(inplace= True))
        self.max_pool = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1, dilation=1)
        
        self.block1 = DenseBlock(self.inner_channels,n_layers[0],self.growth_rate)
        self.inner_channels = self.block1.inner_channels
        out_channels= int(self.inner_channels * self.compression)
        self.transition1 = TransitionBlock(self.inner_channels, out_channels)
        self.inner_channels = out_channels
        
        self.block2 = DenseBlock(self.inner_channels,n_layers[1],self.growth_rate)
        self.inner_channels = self.block2.inner_channels
        out_channels= int(self.inner_channels * self.compression)
        self.transition2 = TransitionBlock(self.inner_channels, out_channels)
        self.inner_channels = out_channels
        
        self.block3 = DenseBlock(self.inner_channels,n_layers[2],self.growth_rate)
        self.inner_channels = self.block3.inner_channels
        out_channels= int(self.inner_channels * self.compression)
        self.transition3 = TransitionBlock(self.inner_channels, out_channels)
        self.inner_channels = out_channels

        self.block4 = DenseBlock(self.inner_channels,n_layers[3],self.growth_rate)        
        self.inner_channels = self.block4.inner_channels
        
        self.norm5 = nn.BatchNorm2d(self.inner_channels)
        self.averag_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.inner_channels, self.output_dim, bias=True)
                
        

        
        
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        x = self.transition3(x)
        x = self.block4(x)
        x = self.norm5(x)
        x = self.averag_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
    
        return x




densenet121_config = [6,12,24,16]

densenet169_config = [6,12,32,32]

densenet201_config = [6,12,48,32]

densenet264_config = [6,12,64,48]



def densenet121(output_dim: int = 1000, 
                image_channels :int = 3
               ) -> DenseNet:
    
    return DenseNet(config=densenet121_config, 
                    output_dim=output_dim, 
                    image_channels=image_channels
                    )



def densenet169(output_dim: int = 1000,
                image_channels :int = 3
               ) -> DenseNet:

    return DenseNet(config=densenet169_config, 
                    output_dim=output_dim, 
                    image_channels=image_channels
                   )



def densenet201(output_dim: int = 1000,
                image_channels: int = 3
               ) -> DenseNet:
    
    return DenseNet(config=densenet201_config, 
                    output_dim=output_dim, 
                    image_channels=image_channels
                   )

def densenet264(output_dim: int = 1000,
                image_channels: int = 3
               ) -> DenseNet:
    
    return DenseNet(config=densenet264_config,
                    output_dim=output_dim, 
                    image_channels=image_channels
                   )