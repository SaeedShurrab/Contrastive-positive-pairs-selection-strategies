import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,**kwargs):
        super(ConvBlock,self).__init__()
        
        self.block = nn.Sequential(nn.Conv2d(**kwargs),
                                   nn.BatchNorm2d(kwargs['out_channels'],eps=0.001),
                                   nn.ReLU(inplace= True))
    
    def forward(self,x):
        x = self.block(x)
        return x


class InceptionA(nn.Module):
    def __init__(self,in_channels,pool_channels):
        super(InceptionA,self).__init__()
        
        self.branch1x1 = ConvBlock(in_channels= in_channels, out_channels= 64,kernel_size= 1, bias = False)
        
        self.branch5x5 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 48, kernel_size= 1,
                                                 bias= False),
                                       ConvBlock(in_channels= 48, out_channels= 64, kernel_size= 5, padding= 2,
                                                 bias= False))
        
        self.branch3x3 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 64, kernel_size= 1,
                                                 bias= False),
                                       ConvBlock(in_channels= 64, out_channels= 96, kernel_size= 3, padding= 1,
                                                 bias= False),
                                       ConvBlock(in_channels= 96, out_channels= 96, kernel_size= 3, padding= 1, 
                                                 bias= False))
        
        self.poolbranch = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
                                        ConvBlock(in_channels= in_channels, out_channels= pool_channels,
                                                  kernel_size= 1, bias= False))
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch5x5(x)
        branch3 = self.branch3x3(x)
        branch4 = self.poolbranch(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        
        x = torch.cat(outputs, 1)
        return x



class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB,self).__init__()
        
        self.branch3x3_1 = ConvBlock(in_channels = in_channels, out_channels= 384, kernel_size= 3, stride= 2,
                                   bias= False)
        
        self.branch3x3_2 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 64, kernel_size= 1,
                                         bias= False),
                                         ConvBlock(in_channels= 64, out_channels= 96, kernel_size= 3, padding= 1,
                                                   bias= False),
                                         ConvBlock(in_channels= 96, out_channels= 96, kernel_size= 3, stride= 2,
                                                  bias= False))
        
        self.poolbranch = nn.MaxPool2d(kernel_size=3, stride=2)
        
    
    def forward(self, x):
        branch1 = self.branch3x3_1(x)
        branch2 = self.branch3x3_2(x)
        branch3 = self.poolbranch(x)
        
        outputs = [branch1, branch2, branch3]
        
        x = torch.cat(outputs,1)
        
        return x


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        
        self.branch1x1 = ConvBlock(in_channels= in_channels, out_channels= 192, kernel_size= 1, bias=  False)
        
        c7 = channels_7x7
        self.branch7x7_1 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= c7, kernel_size= 1,
                                                   bias= False),
                                         ConvBlock(in_channels= c7, out_channels= c7, kernel_size= (7, 1),
                                                   padding= (3,0), bias= False),
                                         ConvBlock(in_channels= c7, out_channels= c7, kernel_size= (1, 7),
                                                   padding= (0,3), bias= False),
                                         ConvBlock(in_channels= c7, out_channels= c7, kernel_size= (7, 1),
                                                   padding= (3,0), bias= False),
                                         ConvBlock(in_channels= c7, out_channels= 192, kernel_size= (1, 7),
                                                   padding= (0,3), bias= False)) 
        
        self.branch7x7_2 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= c7, kernel_size= 1,
                                                  bias = False),
                                         ConvBlock(in_channels= c7, out_channels= c7, kernel_size= (1, 7),
                                                   padding=(0, 3), bias= False),
                                         ConvBlock(in_channels= c7, out_channels= 192, kernel_size= (7, 1),
                                                   padding= (3,0), bias = False))
        
        self.poolbranch = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                        ConvBlock(in_channels= in_channels, out_channels= 192, kernel_size= 1,
                                                 bias= False))
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch7x7_1(x)
        branch3 = self.branch7x7_2(x)
        branch4 = self.poolbranch(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        x = torch.cat(outputs, 1)
        return x


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        
        self.branch3x3 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 192, kernel_size= 1,
                                                 bias = False),
                                       ConvBlock(in_channels= 192, out_channels=320, kernel_size= 3, stride= 2,
                                                 bias= False))
        
        
        self.branch7x7 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 192, kernel_size= 1,
                                                 bias= False),
                                       ConvBlock(in_channels= 192,out_channels= 192, kernel_size=(1, 7),
                                                 padding=(0, 3), bias= False),
                                       ConvBlock(in_channels= 192,out_channels= 192, kernel_size=(7, 1),
                                                 padding=(3, 0), bias= False),
                                       ConvBlock(in_channels= 192, out_channels= 192, kernel_size= 3, stride= 2,
                                                 bias= False))
        
        self.poolbranch = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
    def forward(self, x):
        branch1 = self.branch3x3(x)
        branch2 = self.branch7x7(x)
        branch3 = self.poolbranch(x)
        
        outputs = [branch1, branch2, branch3]
        
        x = torch.cat(outputs, 1)
        return x


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        
        self.branch1x1 = ConvBlock(in_channels= in_channels, out_channels= 320, kernel_size= 1, bias= False)
        
        self.branch3x3_1 = ConvBlock(in_channels= in_channels, out_channels= 384, kernel_size= 1, bias= False)
        self.branch3x3_1_1 = ConvBlock(in_channels= 384, out_channels= 384, kernel_size= (1, 3), padding= (0, 1),
                                      bias= False)
        self.branch3x3_1_2 = ConvBlock(in_channels= 384, out_channels= 384, kernel_size= (3, 1), padding= (1, 0),
                                      bias= False)
        
        
        self.branch3x3_2 = nn.Sequential(ConvBlock(in_channels= in_channels, out_channels= 448, kernel_size= 1,
                                                   bias= False),
                                         ConvBlock(in_channels= 448, out_channels= 384, kernel_size=3, padding=1,
                                                   bias= False))
        self.branch3x3_2_1 = ConvBlock(in_channels= 384, out_channels= 384, kernel_size= (1, 3), padding= (0, 1),
                                       bias = False) 
        self.branch3x3_2_2 = ConvBlock(in_channels= 384, out_channels= 384, kernel_size= (3, 1), padding= (1, 0),
                                       bias = False)         
        
        self.poolbranch = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                        ConvBlock(in_channels= in_channels, out_channels= 192, kernel_size= 1,
                                                 bias= False))
        
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        
        branch2 = self.branch3x3_1(x)
        branch2_1 = self.branch3x3_1_1(branch2)
        branch2_2 = self.branch3x3_1_2(branch2)
        cache_br_2 = [branch2_1, branch2_2]
        branch2 = torch.cat(cache_br_2, 1)
        
        branch3 = self.branch3x3_2(x)
        branch3_1 = self.branch3x3_2_1(branch3)
        branch3_2 = self.branch3x3_2_2(branch3)
        cache_br_3 = [branch3_1,branch3_2]
        branch3 = torch.cat(cache_br_3,1)
        
        branch4 = self.poolbranch(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        
        x = torch.cat(outputs, 1)
        
        return x

class InceptionAux(nn.Module):
    def __init__(self,in_channels, output_dim):
        super(InceptionAux,self).__init__()

        
        self.conv1 = ConvBlock(in_channels= in_channels, out_channels= 128, kernel_size= 1, bias= False)
        self.conv2 = ConvBlock(in_channels= 128, out_channels= 768, kernel_size= 5, bias= False)
        
        self.avg_pool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        
        self.avg_pool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.conv2.stddev = 0.01
        

        self.fc1 = nn.Linear(in_features= 768,out_features= output_dim)
        self.fc1.stddev = 0.001 

        
    def forward(self, x):
        x = self.avg_pool1(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        
        x = torch.flatten(x,1)
        x = self.fc1(x)

        
        return x
        
        

class InceptionV3(nn.Module):
    def __init__(self,image_channels, output_dim= 1000, clf= True, aux_clf=True):
        super(InceptionV3, self).__init__()
        
        self.aux_clf= aux_clf
        self.clf = clf
        
        self.conv1 = ConvBlock(in_channels= image_channels, out_channels= 32, kernel_size= 3,
                               stride= 2,bias = False)
        self.conv2 = ConvBlock(in_channels= 32, out_channels= 32, kernel_size= 3, bias= False)
        self.conv3 = ConvBlock(in_channels= 32, out_channels= 64, kernel_size= 3, padding= 1, bias= False)
        self.max_pool1 = nn.MaxPool2d(kernel_size= 3, stride=2)
        self.conv4 = ConvBlock(in_channels= 64, out_channels= 80, kernel_size= 1, bias= False)
        self.conv5 = ConvBlock(in_channels= 80, out_channels= 192, kernel_size= 3, bias= False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
        self.inception_a_1 = InceptionA(in_channels= 192, pool_channels= 32)
        self.inception_a_2 = InceptionA(in_channels= 256, pool_channels= 64)
        self.inception_a_3 = InceptionA(in_channels= 288, pool_channels= 64)
        
        self.inception_b_1 = InceptionB(288)
        
        self.inception_c_1 = InceptionC(in_channels= 768, channels_7x7= 128)
        self.inception_c_2 = InceptionC(in_channels= 768, channels_7x7= 160)
        self.inception_c_3 = InceptionC(in_channels= 768, channels_7x7= 160)
        self.inception_c_4 = InceptionC(in_channels= 768, channels_7x7= 192)
        
        if self.aux_clf:
            self.aux = InceptionAux(in_channels=768, output_dim= output_dim)
        
        self.inception_d_1 = InceptionD(in_channels= 768)
        
        self.inception_e_1 = InceptionE(in_channels= 1280)
        self.inception_e_2 = InceptionE(in_channels= 2048)
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(2048,output_dim) 
        
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool2(x)
        
        x = self.inception_a_1(x)
        x = self.inception_a_2(x)
        x = self.inception_a_3(x)
        
        x = self.inception_b_1(x)
        
        x = self.inception_c_1(x) 
        x = self.inception_c_2(x) 
        x = self.inception_c_3(x) 
        x = self.inception_c_4(x)
        
        if self.aux_clf:
            x_aux = self.aux(x)
        
        
        x = self.inception_d_1(x)
        
        x = self.inception_e_1(x)
        x = self.inception_e_2(x)
        x = self.avg_pool1(x)

        if self.clf:
            x = self.dropout(x)
            x = torch.flatten(x,1)
            x = self.fc1(x)
        
        if self.aux_clf:
            return x, x_aux
        else:
            return x
        
        

        
        

def inceptionv3(imag_channels:int = 3, output_dim:int = 1000,clf:bool =True, aux_clf: bool =True) -> InceptionV3:
    return InceptionV3(image_channels= imag_channels, output_dim= output_dim, clf= clf, aux_clf= aux_clf)