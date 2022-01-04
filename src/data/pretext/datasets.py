import os
from typing import Optional, Tuple, List, Iterable
import torch
from torch import Tensor
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


defualt_transforms = T.Compose([T.Resize((300,300)),
                                T.RandomApply([T.RandomRotation(degrees=(10)), 
                                               T.RandomAffine(degrees=0, shear=20, scale=(1,1))], p=1),
                                T.RandomHorizontalFlip(p=0.5),
                                T.RandomApply([T.ColorJitter(brightness=0.5, hue=.3),
                                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5))], p= 5),
                                T.ToTensor(),
                                T.Normalize(mean=torch.tensor([0.1123,0.1123,0.1123]),
                                            std=torch.tensor([0.1228,0.1228,0.1228])) 
                        ])


specific_transforms = T.Compose([T.Resize((300,300)),
                                T.ToTensor(),
                                T.Normalize(mean=torch.tensor([0.1123,0.1123,0.1123]),
                                            std=torch.tensor([0.1228,0.1228,0.1228])) 
                        ])

class UnrestrictedOCT(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transforms: Optional[T.Compose]
                ) -> None:
        
        self.transforms = transforms
        self.all_images = self._get_all_images(data_dir)

        if self.transforms is None:
            self.transforms = defualt_transforms
            

    def __getitem__(self, 
                    idx: int
                   ) -> Tuple[Tensor]:
        
        image = self.all_images[idx]
        view1 = self.transforms(Image.open(fp=image).convert('RGB'))
        view2 = self.transforms(Image.open(fp=image).convert('RGB'))
        return (view1, view2)
        
        
    def __len__(self) -> int:
        return len(self.all_images)
    
    
    def _get_all_images(self, 
                        data_dir
                       ) -> List[List[str]]:
        
        all_images = []
        patients = os.listdir(data_dir)
        
        for patient in patients:
            sessions = os.listdir(os.path.join(data_dir,patient))
            
            for session in sessions:
                eye_directions = os.listdir(os.path.join(data_dir,patient,session))
                
                for direction in eye_directions:
                    images =  os.listdir(os.path.join(data_dir,patient,session,direction))
                    
                    for image in images:
                        all_images.append(os.path.join(data_dir,patient,session,direction,image)) 
        
        return all_images



class UnrestrictedDataModule(pl.LightningDataModule): 
    def __init__(self,
                data_dir: str,
                transforms: Optional[T.Compose],
                batch_size: int = 128,
                num_workers: int = 8,
                pin_memory: bool = True 
                ) -> None:
        super(UnrestrictedDataModule,self).__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str]):
        self.train_dataset = UnrestrictedOCT(data_dir=self.data_dir, transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )



#---------------------------------------------------------------------


class XYRetinaOCT(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transforms: Optional[T.Compose]
                ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.all_pairs = self._get_all_pairs(data_dir)
        
        
        if self.transforms is None:
            self.transforms = specific_transforms


    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        view1, view2 = self.all_pairs[idx]
        view1 = self.transforms(Image.open(fp=view1).convert('RGB'))
        view2 = self.transforms(Image.open(fp=view2).convert('RGB'))
        return (view1, view2)
        
        
    def __len__(self) -> int:
        return len(self.all_pairs)
    
    
    def _get_all_pairs(self, 
                       data_dir: str
                      ) -> List[List[str]]:
        
        all_pairs = []
        patients = os.listdir(data_dir)
        
        for patient in patients:
            sessions = os.listdir(os.path.join(data_dir,patient))
            
            for session in sessions:
                eye_directions = os.listdir(os.path.join(data_dir,patient,session))
                
                for direction in eye_directions:
                    images =  os.listdir(os.path.join(data_dir,patient,session,direction))
                    temp_pairs = []
                    
                    if len(images) == 2:
                        for image in images:
                            temp_pairs.append(os.path.join(data_dir,patient,session,direction,image))
                        all_pairs.append(temp_pairs)
                        
                    elif len(images) < 2:
                        continue
                        
                    elif len(images) > 2 and len(images) %2 == 0:
                        x ,y = [],[]
                        for image in images:
                            x.append(image) if image.endswith('x.bmp') else y.append(image)
                            
                        x.sort()
                        y.sort()
                        
                        for img1, img2 in zip(x,y):
                            temp = [os.path.join(data_dir,patient,session,direction,img1),
                                    os.path.join(data_dir,patient,session,direction,img2)]
                            all_pairs.append(temp)      
        return all_pairs


class XYScansDataModule(pl.LightningDataModule): 
    def __init__(self,
                data_dir: str,
                transforms: Optional[T.Compose],
                batch_size: int = 128,
                num_workers: int = 8,
                pin_memory: bool = True 
                ) -> None:
        super(XYScansDataModule, self).__init__()
        
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str]):
        self.train_dataset = XYRetinaOCT(data_dir=self.data_dir, transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )



#---------------------------------------------------------------------

class ConsecutiveSessionsOCT(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transforms: Optional[T.Compose]
                ) -> None:
        
        self.transforms = transforms
        self.all_pairs = self._get_all_pairs(data_dir)
        
        if self.transforms is None:
            self.transforms = specific_transforms

            
    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        view1, view2 = self.all_pairs[idx]
        view1 = self.transforms(Image.open(fp=view1).convert('RGB'))
        view2 = self.transforms(Image.open(fp=view2).convert('RGB'))
        return (view1, view2)
        
        

    def __len__(self) -> int:
        return len(self.all_pairs)

    
    def consecutive_pairs_generator(self,
                                    input_list: Iterable
                                    ) -> List[List[int]]:
        input_list.sort()
        output_list = []

        for item in range(len(input_list) - 1): 
            output_list.append([input_list[item],input_list[item + 1]])
        
        return output_list
    
    
    def _get_all_pairs(self, 
                       data_dir: str
                      ) -> List[List[str]]:
        all_pairs = []
        patients = os.listdir(data_dir)
        for patient in patients:        
            sessions = os.listdir(os.path.join(data_dir,patient))

            if len(sessions) <= 1:
                continue
            
            sessions_pairs  = self.consecutive_pairs_generator(sessions)
    
            for session in sessions_pairs:
                session1 = session[0]
                session2 = session[1]
        
                session1_directions = sorted(os.listdir(os.path.join(data_dir,patient,session[0])))
                session2_directions = sorted(os.listdir(os.path.join(data_dir,patient,session[1])))
        
                session1_l = session1_directions[0]
                session1_r = session1_directions[1]
        
                session2_l = session2_directions[0]
                session2_r = session2_directions[1]
        
                session1_l_images = os.listdir(os.path.join(data_dir,patient,session1,session1_l))
                session1_r_images = os.listdir(os.path.join(data_dir,patient,session1,session1_r))

                session2_l_images = os.listdir(os.path.join(data_dir,patient,session2,session2_l))
                session2_r_images = os.listdir(os.path.join(data_dir,patient,session2,session2_r))
        
                for image1 in session1_l_images:
                    for image2 in session2_l_images:
                        if image1.endswith('x.bmp') and image2.endswith('x.bmp'):
                            all_pairs.append([os.path.join(os.path.join(data_dir,patient,
                                                                        session1,session1_l, image1)),
                                              os.path.join(os.path.join(data_dir,patient,
                                                                        session2,session2_l, image2))
                                             ])
                                    
                for image1 in session1_r_images:
                    for image2 in session2_r_images:
                        if image1.endswith('x.bmp') and image2.endswith('x.bmp'):
                            all_pairs.append([os.path.join(os.path.join(data_dir,patient,
                                                                        session1,session1_r, image1)),
                                              os.path.join(os.path.join(data_dir,patient,
                                                                        session2,session2_r, image2))
                                             ])
                        
                for image1 in session1_l_images:
                    for image2 in session2_l_images:
                        if image1.endswith('y.bmp') and image2.endswith('y.bmp'):
                            all_pairs.append([os.path.join(os.path.join(data_dir,patient,
                                                                        session1,session1_l, image1)),
                                              os.path.join(os.path.join(data_dir,patient,
                                                                        session2,session2_l, image2))
                                             ])

                for image1 in session1_r_images:
                    for image2 in session2_r_images:
                        if image1.endswith('y.bmp') and image2.endswith('y.bmp'):
                            all_pairs.append([os.path.join(os.path.join(data_dir,patient,
                                                                        session1,session1_r, image1)),
                                              os.path.join(os.path.join(data_dir,patient,
                                                                        session2,session2_r, image2))
                                             ])
        return all_pairs



class ConsicutiveSessionsDataModule(pl.LightningDataModule): 
    def __init__(self,
                data_dir: str,
                transforms: Optional[T.Compose],
                batch_size: int = 128,
                num_workers: int = 8,
                pin_memory: bool = True 
                ) -> None:
        super(ConsicutiveSessionsDataModule, self).__init__()
        
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str]):
        self.train_dataset = ConsecutiveSessionsOCT(data_dir=self.data_dir, transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory= self.pin_memory
                         )