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
                             T.Normalize(mean=torch.tensor([0.1123,0.1123,0.1123]), 
                                         std=torch.tensor([0.1228,0.1228,0.1228])) 
                        ])


val_test_transform = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                T.Normalize(mean=torch.tensor([0.1123,0.1123,0.1123]), 
                                             std=torch.tensor([0.1228,0.1228,0.1228])), 
                        ])




class ClassificationDataset(Dataset):
    def __init__(self,
                 data_dir:  str,
                 form: str = 'binary',
                 transform: Optional[T.Compose] = None,
                 ) -> None:
        

        self.data_dir = data_dir
        self.form = form
        self.transform = transform
        self.all_images = self._get_images(self.data_dir)
        
    def __len__(self):
        return len(self.all_images)
    
    
    
    def __getitem__(self,
                    idx: int
                    ) -> Tuple[Tensor,int]:
        
        image = self.all_images[idx]


        if self.form == 'binary':
            label = torch.tensor([0]) if image.split('/')[-2] == 'Normal' else torch.tensor([1])
            image = self.transform(Image.open(image))
        
        elif self.form == 'multi-class':
            if image.split('/')[-2] == 'Normal':
                label = torch.tensor([0])
                image = self.transform(Image.open(image)) 
                
            elif image.split('/')[-2] == 'CNV':
                label = torch.tensor([1])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'CSR':
                label = torch.tensor([2])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'GA':
                label = torch.tensor([3])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'MRO':
                label = torch.tensor([4])
                image = self.transform(Image.open(image))  

            elif image.split('/')[-2] == 'VMT':
                label = torch.tensor([5])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'MH':
                label = torch.tensor([6])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'FMH':
                label = torch.tensor([6])
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'PMH':
                label = torch.tensor([7])
                image = self.transform(Image.open(image))  

                
        elif self.form == 'grading':
            if image.split('/')[-2] == 'mild':
                label = torch.tensor(0)
                image = self.transform(Image.open(image)) 
                
            elif image.split('/')[-2] == 'moderate':
                label = torch.tensor(1)
                image = self.transform(Image.open(image))  
                
            elif image.split('/')[-2] == 'severe':
                label = torch.tensor(2)
                image = self.transform(Image.open(image))
        
        
        return (image,label)
        

        
    
    def _get_images(self,
                    data_dir: str
                   ) -> List[str]: 
    
        labels = os.listdir(data_dir)
        all_images = []
    
        for label in labels:
            images = os.listdir(os.path.join(data_dir,label))
        
            for image in images:
                all_images.append(os.path.join(data_dir,label,image))
    
        return all_images



class DownStreamDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 form:str = 'grading',
                 training_transforms: Optional[T.Compose] = None,
                 val_test_transforms: Optional[T.Compose] = None,
                 batch_size: int = 128,
                 num_workers: int = 8,
                 pin_memory: bool = True 
                ) -> None:
        super(DownStreamDataModule,self).__init__()
        self.data_dir = data_dir
        self.form = form
        self.training_transforms = training_transforms
        self.val_test_transforms = val_test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        if training_transforms is None:
            self.training_transforms = train_transform

        if val_test_transforms is None:
            self.val_test_transforms = val_test_transform


    def setup(self, stage: Optional[str]):
        self.train_dataset = ClassificationDataset(data_dir=os.path.join(self.data_dir,'train'),
                                                   form=self.form,
                                                   transform=self.training_transforms
                                                  )
        self.val_dataset = ClassificationDataset(data_dir=os.path.join(self.data_dir,'val'), 
                                                 form=self.form,
                                                 transform=self.val_test_transforms
                                                )
        self.test_dataset = ClassificationDataset(data_dir=os.path.join(self.data_dir,'test'), 
                                                  form=self.form,
                                                  transform=self.val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory,
                         )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset,batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset,batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )