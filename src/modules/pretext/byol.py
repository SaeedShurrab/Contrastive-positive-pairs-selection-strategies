from typing import Dict, Optional, List, Any

from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import glob
from ...models.sslmodels.byol import BYOL




class ByolModel(pl.LightningModule):
    def __init__(self,
                backbone: nn.Module,
                criterion: nn.Module ,
                target_decay: float = 0.996,
                optimizer: str = 'sgd',
                learning_rate: float =  1e-2,
                weight_decay: float = 0.0,
                scheduler: Optional[str] = 'step',
                sched_step_size: int = 5,
                sched_gamma: float = 0.5,
                max_epochs: int =100
                ) -> None:
        super(ByolModel, self).__init__()

        self.save_hyperparameters()
        self.backbone = backbone()
        self.learner = BYOL(model=self.backbone, target_decay=target_decay)
        self.criterion = criterion()
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.max_epochs = max_epochs

    def training_step(self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        view1, view2 = batch
        v1_on, v2_tar, v2_on, v1_tar = self.learner(view1,view2)
        loss = self.criterion(v1_on, v2_tar, v2_on, v1_tar)
        
        self.log("train_loss", loss, on_epoch= True,on_step=True ,prog_bar=True, logger=True)
        #self.logger.experiment.log_artifact()
        return loss

    def on_fit_start(self) -> None:
        self.logger.experiment.log_artifact(self.logger.run_id,'./args.json')
    
    def on_before_zero_grad(self, _) -> None:
        self.learner.update_target_network()

    def on_fit_start(self) -> None:
        self.logger.experiment.log_artifact('/src/args.json')
    

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.learner.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(params=self.learner.parameters(), 
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