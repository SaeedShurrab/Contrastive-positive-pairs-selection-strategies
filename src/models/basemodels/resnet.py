
import torch 
import torch.nn as nn

from collections import namedtuple

from torch import Tensor
from typing import Callable, Any, NamedTuple, Optional, Tuple, List



class ConvBlock(nn.Module):
    """Convolutional layer along with batch normalization layer in a single block
        
    Basic args:
        in_channels - int: input channels
        out_channels - int: output channels
        kenel_size - int or tuple: applied filter size
        padding - int or tuple: applied padding magnitude
        stride - int or tuole: filter movement over the input image
        bias - bool: bias parameter inclusion or exclusion  
        
        for more info: see nn.Conv2d and nn.BatchNorm2d documentation
    """        

    def __init__(self,
                 **kwargs
                ) -> None:   
        super(ConvBlock,self).__init__()
        
        self.block = nn.Sequential(nn.Conv2d(**kwargs,),
                                   nn.BatchNorm2d(kwargs['out_channels']))
    
    def forward(self,
                x: Tensor
                ) -> Tensor:
        x = self.block(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1 ,
                 downsample: bool = False
                ) -> None:
        super(BasicBlock,self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                kernel_size=3, padding=1, bias=False)
        
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=1,
                                kernel_size=3, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            self.downsample = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, stride=2, bias=False)
        else:
            self.downsample = None
            
        
    def forward(self, 
                x: Tensor
               ) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
            
        if self.downsample != None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
            
        return x


class Bottleneck(nn.Module):
    
    expansion = 4
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 downsample: bool = False
                ) -> None:
        super(Bottleneck,self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=1,
                               kernel_size=1, bias=False)
        
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)
        
        self.conv3 = ConvBlock(in_channels=out_channels, out_channels=out_channels * self.expansion, 
                               stride=1, kernel_size=1, bias=False)
        
        self.relu = nn.ReLU(inplace= True)
        
        
        if downsample:
            self.downsample = ConvBlock(in_channels= in_channels, out_channels= out_channels * self.expansion,
                                        kernel_size=1, stride= stride, bias= False)
        else:
            self.downsample = None
            
    
    def forward(self,
                x: Tensor
               ) -> Tensor:
        
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        
        if self.downsample != None:
            identity = self.downsample(identity)
            
        x += identity
        x = self.relu(x)
        
        return x
    

class ResNet(nn.Module):
    def __init__(self,
                 config: NamedTuple , 
                 output_dim: int = 10, 
                 image_channels: int = 1
                ) -> None:
        super(ResNet, self).__init__()
    
        block, n_blocks, channels = config
        self.image_channels= image_channels
        self.output_dim = output_dim
        self.in_channels = channels[0]
    
        assert len(n_blocks) == len(channels) == 4
    
        self.conv1 = ConvBlock(in_channels= self.image_channels, out_channels= self.in_channels,kernel_size= 7, stride=2,
                           padding= 3, bias= False)
        self.relu = nn.ReLU(inplace= True)
        self.max_pool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding=1)
        
    
        self.conv2_x = self._make_layer(block, n_blocks[0],channels[0])
        self.conv3_x = self._make_layer(block, n_blocks[1],channels[1], stride= 2)
        self.conv4_x = self._make_layer(block, n_blocks[2],channels[2], stride= 2)
        self.conv5_x = self._make_layer(block, n_blocks[3],channels[3], stride= 2)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, self.output_dim)
    
    
    
    def forward(self, 
                x : Tensor
               ) -> Tensor:
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    
    
    def _make_layer(self,
                    block: Callable, 
                    n_blocks: int, 
                    channels: int, 
                    stride: int = 1
                   ) -> nn.Module:
        
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
            
        layers.append(block(self.in_channels,channels,stride,downsample))
        
        for i in range(1,n_blocks):
            layers.append(block(block.expansion * channels, channels))
            
        self.in_channels = block.expansion * channels
        
        return nn.Sequential(*layers)
    
    



ResNetConfig = namedtuple('ResNetConfig', 
                         ['block', 'n_blocks', 'channels']
                         )


resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2,2,2,2],
                               channels = [64, 128, 256, 512])

resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3,4,6,3],
                               channels = [64, 128, 256, 512])

resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

resnet101_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 23, 3],
                                channels = [64, 128, 256, 512])

resnet152_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])



def resnet18(output_dim: int = 1000,
             image_channels: int =1,
            ) -> ResNet:

    config = resnet18_config
    
    return ResNet(config, 
                  output_dim=output_dim, 
                  image_channels=image_channels
                  )



def resnet34(output_dim: int = 1000,
             image_channels: int = 3,
            ) -> ResNet:

    config = resnet34_config

    return ResNet(config,
                  output_dim=output_dim,
                  image_channels=image_channels
                  )



def resnet50(output_dim: int = 1000,
             image_channels: int = 3, 
            ) -> ResNet:

    config = resnet50_config

    return ResNet(config,
                  output_dim=output_dim,
                  image_channels=image_channels
                  )



def resnet101(output_dim: int = 1000,
              image_channels: int = 3, 
             ) -> ResNet:
    
    config = resnet101_config
    
    return ResNet(config, 
                  output_dim=output_dim,
                  image_channels=image_channels
                  )



def resnet152(output_dim: int = 1000,
              image_channels: int = 3,
             ) -> ResNet:
    
    config = resnet152_config
    
    return ResNet(config,
                  output_dim=output_dim, 
                  image_channels=image_channels
                  )