import os
from PIL import Image
from pytorch_lightning import loggers
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

import pytorch_lightning as pl


transforms = T.Compose([T.Resize((32,32)),
                                T.ToTensor(),

                        ])

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
        x = self.fc1(x)
        return x


print(os.getcwd)
class SimpleDataset(Dataset):
    def __init__(self,n_samples = 50,
                 ):
        self.n_samples = n_samples
        self.images = torch.randn(n_samples,3,224,224)
        self.labels = torch.randint(0,2, (1, n_samples))

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[:,index]
        return image, label

    def __len__(self):
        return self.images.shape[0]



class SimpleDataLoader(pl.LightningDataModule): 
    def __init__(self,
                n_samples = 100,
                batch_size: int = 8,
                ) -> None:
        super(SimpleDataLoader,self).__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage = None):
        self.train_dataset = ImageFolder(root='/src/src/models/basemodels/data/train',transform =transforms)
        self.val_dataset = ImageFolder(root='/src/src/models/basemodels/data/val', transform=transforms)

        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=False
                         )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset,batch_size=self.batch_size,
                          shuffle=False
                         )


class SimpleModel(pl.LightningModule):
    def __init__(self, ):
        super(SimpleModel,self).__init__()
        self.model = SimpleNet()
        self.criterion = nn.NLLLoss(reduction="sum")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y,y_hat)
        self.log('train_loss',loss, prog_bar=True,logger=True, on_epoch=True, on_step=True )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y,y_hat)
        self.log('val_loss',loss, prog_bar=True,logger=True, on_epoch=True, on_step=True )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


