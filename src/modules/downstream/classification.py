from typing import Optional, List

from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchmetrics.functional import  accuracy, precision,recall, specificity





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
                max_epochs: int = 50
                ) -> None:
        super(ClassificationModel, self).__init__()

        self.save_hyperparameters()
        self.model = model()
        self.criterion = criterion()
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.max_epochs = max_epochs

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = nn.Linear(self.model.fc.in_features,output_dim)

    def forward(self,
                x: Tensor
               ) -> Tensor:
        return self.model(x)

    def training_step(self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        input, label = batch
        prediction = self.model(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)
        
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True)
        self.log("train_acc", acc, on_epoch= True,on_step=True , logger=True)
        return loss

    
    def validation_step (self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        input,label = batch
        prediction = self.model(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)

        self.log("val_loss", loss, on_epoch= True,on_step=True , logger=True)
        self.log("val_acc", acc, on_epoch= True,on_step=True , logger=True)
        return loss


    def test_step(self, 
                  batch: List[Tensor], 
                  batch_idx: int
                 ) -> float:
        input, label = batch
        prediction = self.model(input)
        loss = self.criterion(prediction, label)
        acc = accuracy(preds=prediction, target=label)

        self.log("test_loss", loss, on_epoch= True,on_step=True , logger=True)
        self.log("test_acc", acc, on_epoch= True,on_step=True , logger=True)
        return loss 

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






