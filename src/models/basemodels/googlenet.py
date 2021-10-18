

import torch 
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,**kwargs):
        super(ConvBlock,self).__init__()
        
        self.block = nn.Sequential(nn.Conv2d(**kwargs),
                                   nn.BatchNorm2d(kwargs['out_channels']),
                                   nn.ReLU(inplace= True))
    
    def forward(self,x):
        x = self.block(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red,ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        
        
        self.branch1 = ConvBlock(in_channels= in_channels, out_channels= ch1x1, kernel_size= 1, bias= False)
        
        self.branch2 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= ch3x3red, 
                                               kernel_size= 1, bias= False),
                                     ConvBlock(in_channels= ch3x3red, out_channels= ch3x3, kernel_size= 3,
                                               padding= 1, bias= False))
        # remember the kernel size and padding 
        self.branch3 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= ch5x5red, 
                                               kernel_size= 1, bias= False),
                                     ConvBlock(in_channels= ch5x5red, out_channels= ch5x5, kernel_size= 5,
                                               padding= 2, bias= False))
        
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size= 3, padding= 1 , stride= 1,ceil_mode= True),
                                    ConvBlock(in_channels= in_channels, out_channels= pool_proj,
                                              kernel_size= 1, bias= False))
        
    def forward (self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branches = [branch1,branch2, branch3, branch4]
        x =  torch.cat(branches,1)
        return x



class InceptionAux(nn.Module):
    def __init__(self,in_channels, output_dim, aux_clf):
        super(InceptionAux,self).__init__()
        self.aux_clf= aux_clf
        
        self.conv1 = ConvBlock(in_channels= in_channels, out_channels= 128, kernel_size= 1, bias= False)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4,4))
        
        self.fc1 = nn.Linear(in_features= 2048, out_features= 1024)
        self.fc2 = nn.Linear(in_features=1024,out_features= output_dim)
        
        self.dropout = nn.Dropout(p= 0.7)
        self.relu = nn.ReLU(inplace= True)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        if self.aux_clf:
            x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
        
class GoogleNet(nn.Module):
    def __init__(self,image_channels= 3, output_dim=1000, clf= True, aux_clf= True):
        super(GoogleNet,self).__init__()
        

        self.aux_clf = aux_clf
        self.clf = clf
        
        self.conv1 = ConvBlock(in_channels= image_channels, out_channels= 64, kernel_size= 7, 
                               stride= 2, padding= 3, bias= False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride= 2, ceil_mode= True)
        
        self.conv2 = ConvBlock(in_channels = 64, out_channels= 64, kernel_size= 1, bias= False)
        self.conv3 = ConvBlock(in_channels= 64, out_channels= 192, kernel_size= 3, padding= 1, bias= False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride= 2, ceil_mode= True)
        
        
        self.inception3a = InceptionBlock(in_channels= 192, ch1x1= 64, ch3x3red= 96, ch3x3= 128, 
                                          ch5x5red= 16, ch5x5=32, pool_proj=32)
        self.inception3b = InceptionBlock(in_channels= 256, ch1x1= 128, ch3x3red= 128, ch3x3= 192, 
                                          ch5x5red= 32, ch5x5=96, pool_proj=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride= 2, ceil_mode= True)
        
        
        
        self.inception4a = InceptionBlock(in_channels= 480, ch1x1= 192, ch3x3red= 96, ch3x3= 208, 
                                          ch5x5red= 16, ch5x5=48, pool_proj=64)
        self.inception4b = InceptionBlock(in_channels= 512, ch1x1= 160, ch3x3red= 112, ch3x3= 224, 
                                          ch5x5red= 24, ch5x5=64, pool_proj=64)
        self.inception4c = InceptionBlock(in_channels= 512, ch1x1= 128, ch3x3red= 128, ch3x3= 256, 
                                          ch5x5red= 24, ch5x5=64, pool_proj=64)
        self.inception4d = InceptionBlock(in_channels= 512, ch1x1= 112, ch3x3red= 144, ch3x3= 288, 
                                          ch5x5red= 32, ch5x5=64, pool_proj=64)
        self.inception4e = InceptionBlock(in_channels= 528, ch1x1= 256, ch3x3red= 160, ch3x3= 320, 
                                          ch5x5red= 32, ch5x5=128, pool_proj=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride= 2, ceil_mode= True)
        
        
        
        self.inception5a = InceptionBlock(in_channels= 832, ch1x1= 256, ch3x3red= 160, ch3x3= 320, 
                                          ch5x5red= 32, ch5x5=128, pool_proj=128)
        self.inception5b = InceptionBlock(in_channels= 832, ch1x1= 384, ch3x3red= 192, ch3x3= 384, 
                                          ch5x5red= 48, ch5x5=128, pool_proj=128)
        
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p= 0.5)
        self.fc1 = nn.Linear(in_features=1024,out_features=output_dim)
        
        
        if aux_clf:
            self.aux1 = InceptionAux(in_channels= 512, output_dim= output_dim, aux_clf= aux_clf)
            self.aux2 = InceptionAux(in_channels= 528, output_dim= output_dim, aux_clf= aux_clf)
            
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)
        x = self.inception4a(x)
        
        if self.aux_clf:
            x_aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_clf:
            x_aux2 = self.aux2(x)    
        x = self.inception4e(x)
        x = self.max_pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        if self.clf:
            x = torch.flatten(x,1)
            x = self.dropout(x)
            x = self.fc1(x)
        
        if self.aux_clf:
            return x, x_aux1, x_aux2
        else:
            return x



def googlenet(imag_channels:int = 3, output_dim:int = 1000,clf:bool =True, aux_clf: bool =True) -> GoogleNet:
    return GoogleNet(imag_channels,output_dim,clf, aux_clf)