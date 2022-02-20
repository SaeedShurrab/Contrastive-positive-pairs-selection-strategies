from typing import Optional, List

from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchmetrics.functional.classification import  accuracy, precision,recall, specificity, f1
from torchmetrics import Accuracy, Precision


class ClassificationModel(pl.LightningModule):
    def __init__(self,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: str = 'sgd',
                learning_rate: float =  1e-2,
                weight_decay: float = 0.0,
                scheduler: Optional[str] = 'step',
                sched_step_size: int = 5,
                sched_gamma: float = 0.5,
                output_dim: int = 2,
                freeze: bool = False,
                imagenet: bool = False,
                max_epochs: int = 50
                ) -> None:
        super(ClassificationModel, self).__init__()

        self.save_hyperparameters() 
        self.model = model(pretrained=imagenet,num_classes = output_dim)
        self.criterion = criterion
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.max_epochs = max_epochs
        self.output_dim = output_dim



        if freeze:
            for name, param in self.model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
        
        elif imagenet: 
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct <= 6:
                    for param in child.parameters():
                        param.requires_grad = False
        
    

        #self.model.fc = nn.Linear(self.model.fc.in_features,self.output_dim)

    def forward(self,
                x: Tensor
               ) -> Tensor:
        return self.model(x)

    def training_step(self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        input, label = batch
        prediction = self.forward(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)

        
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True)
        self.log("train_acc", acc, on_epoch= True,on_step=True, logger=True)
        return loss

    
    def validation_step (self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        input,label = batch
        prediction = self.forward(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)
        #prec = precision(preds=prediction,target=label,num_classes=self.output_dim, average='weighted')
        #rec = recall(preds=prediction,target=label,num_classes=self.output_dim, average='weighted')
        #spec = specificity(preds=prediction,target=label,num_classes=self.output_dim, average='weighted')
        #f_1 = f1(preds=prediction,target=label,num_classes=self.output_dim, average='weighted')

        self.log("val_loss", loss, on_epoch= True, on_step=True, logger=True)
        self.log("val_acc", acc, on_epoch= True, on_step=True, logger=True)
        #self.log('val_prec',prec, on_epoch=True, logger=True)
        #self.log('val_rec',rec, on_epoch=True, logger=True)
        #self.log('val_spec',spec, on_epoch=True, logger=True)
        #self.log('val_f1',f_1, on_epoch=True, logger=True)
        return loss


    def test_step(self, 
                  batch: List[Tensor], 
                  batch_idx: int
                 ) -> float:
        input, label = batch
        prediction = self.forward(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)
        prec = precision(preds=prediction,target=label,num_classes=self.output_dim, average='micro')
        rec = recall(preds=prediction,target=label,num_classes=self.output_dim, average='micro')
        spec = specificity(preds=prediction,target=label,num_classes=self.output_dim, average='micro')
        f_1 = f1(preds=prediction,target=label,num_classes=self.output_dim, average='micro')

        
    

        self.log("test_loss", loss, on_epoch= True, logger=True)
        self.log("test_acc", acc, on_epoch= True, logger=True)
        self.log('test_prec',prec, on_epoch=True, logger=True)
        self.log('test_rec',rec, on_epoch=True, logger=True)
        self.log('test_spec',spec, on_epoch=True, logger=True)
        self.log('test_f1',f_1, on_epoch=True, logger=True)
        return  loss



    

    def on_fit_start(self) -> None:
        self.logger.experiment.log_artifact(self.logger.run_id,'./args.json')



    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(params=self.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )
        else:
            raise NameError('optimizer must be eithr sgd or adam')


        if self.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                  step_size=self.sched_step_size,
                                                  gamma=self.sched_gamma, verbose=True
                                                 )
        elif self.scheduler == 'exponential':
             scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                          gamma=self.sched_gamma, verbose=True
                                                         )
        elif self.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                            eta_min=0,
                                                            T_max=self.max_epochs
                                                            )


        elif self.scheduler is None:
            scheduler = None

        else: 
            raise NameError('scheduler must be eithr step or exponential')
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
               }






