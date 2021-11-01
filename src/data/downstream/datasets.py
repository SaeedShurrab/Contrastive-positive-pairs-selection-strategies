import os
from typing import Optional, Tuple, List, Iterable
import torch
from torch import Tensor
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl



train_transform = T.Compose([T.Resize((224,224)),
                             T.RandomApply([T.RandomRotation(degrees=(10)), 
                                            T.RandomAffine(degrees=0, shear=10, scale=(1,1))], p=1),
                             T.RandomHorizontalFlip(p=0.5),
                             T.ToTensor(),
                             T.Normalize(mean=torch.tensor([0.1115]), # Rememebr to calculate and update
                                         std=torch.tensor([0.1372])) # Rememebr to calculate and update
                        ])


val_test_transform = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                T.Normalize(mean=torch.tensor([0.1115]), # Rememebr to calculate and update
                                             std=torch.tensor([0.1372])), # Rememebr to calculate and update
                        ])


class DownStreamDataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                train_transforms: Optional[T.Compose],
                val_test_transforms = Optional[T.Compose],
                batch_size: int = 128,
                num_workers: int = 8,
                pin_memory: bool = True 
                ) -> None:
        super(DownStreamDataModule,super).__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        if train_transforms is None:
            self.train_transforms = train_transform

        if val_test_transforms is None:
            self.val_test_transforms = val_test_transform


    def setup(self, stage: Optional[str]):
        self.train_dataset = ImageFolder(root=os.path.join(self.data_dir,'train'), transform=self.train_transforms)
        self.val_dataset = ImageFolder(root=os.path.join(self.data_dir,'val'), transform=self.val_test_transforms)
        self.test_dataset = ImageFolder(root=os.path.join(self.data_dir,'test'), transform=self.val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )