import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch import Tensor
from typing import *

from torch.optim import lr_scheduler


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 projection_dim: int = 256,
                 hidden_dim: int = 4096
                ) -> None:
        super(MLP,self).__init__()
    
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, projection_dim)
        
        )
    
    def forward(self, 
                x: Tensor
               ) -> Tensor:
        
        return self.mlp(x)



class NormalizedMSELoss(nn.Module):
    def __init__(self) -> None:
        super(NormalizedMSELoss,self).__init__()   
        
    def _forward(self, 
                 view1: Tensor , 
                 view2: Tensor
                ) -> Tensor:
        
        view1 = F.normalize(view1, p=2, dim=-1)
        view2 = F.normalize(view2, p=2, dim=-1)
        loss = 2 - 2 * (view1 * view2).sum(dim=-1)
        return loss
    
    def forward(self, 
                v1_online_pred: Tensor,
                v2_target_proj: Tensor,
                v2_online_pred: Tensor,
                v1_target_proj: Tensor
               ) -> Tensor:
        
        loss1 = self._forward(v1_online_pred,v2_target_proj)
        loss2 = self._forward(v2_online_pred,v1_target_proj)
        
        loss = loss1 + loss2
        
        return loss.mean()
        


class EMA:
    def __init__(self,
                target_decay: float
                ) -> None:
        self.target_decay = target_decay
        
        assert 0 <= self.target_decay <= 1, 'target decay must bet between [0-1] inclusive'
        
    def __call__(self,
                 online_weights: Tensor,
                 target_weights: Tensor
                ) -> Tensor:
        
        return target_weights * self.target_decay + (1 - self.target_decay) * online_weights


class EncodProject(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 hidden_dim: int = 4096,
                 projection_out_dim: int = 256
                 ) -> None:
        super(EncodProject, self).__init__()
                
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.projector = MLP(input_dim= nn.Sequential(*list(model.children()))[-1].in_features,
                             projection_dim=projection_out_dim,
                             hidden_dim= hidden_dim)

    def forward(self, 
                x: Tensor
               ) -> Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x


class BYOL(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 hidden_dim: int = 4096,
                 projection_out_dim: int = 256,
                 target_decay: float = 0.99  
                ) -> None:
        super(BYOL, self).__init__()
        
        self.online_encoder = EncodProject(model=model,
                                           hidden_dim=hidden_dim,
                                           projection_out_dim=projection_out_dim
                                          ) 
        self.online_predictor = MLP(input_dim=projection_out_dim,
                                    hidden_dim=hidden_dim,
                                    projection_dim=projection_out_dim
                                   )

        self.target_network = copy.deepcopy(self.online_encoder)
        self._set_requieres_grad(self.target_network, False)
        
        self.moving_average_updater = EMA(target_decay=target_decay)

    
    def _set_requieres_grad(self,
                            model: nn.Module,
                            grad: bool = False
                           ) -> None:
        for param in model.parameters():
            param.requires_grad = grad
          
    @torch.no_grad()    
    def update_target_network(self) -> None:
        for online_weights, target_weights in zip(self.online_encoder.parameters(), 
                                                  self.target_network.parameters()
                                                 ):
            target_weights.data = self.moving_average_updater(online_weights=online_weights.data,
                                                              target_weights=target_weights.data)
            

        
    def forward(self, 
                view1, 
                view2
               ) -> Tuple[Tensor]:
        
        view1_online_projection = self.online_encoder(view1)
        view1_online_prediction = self.online_predictor(view1_online_projection)
        
        view2_online_projection = self.online_encoder(view2)
        view2_online_prediction = self.online_predictor(view2_online_projection)
        
        with torch.no_grad():
            view1_target_projection = self.target_network(view1)        
            view2_target_projection = self.target_network(view2)
            view1_target_projection.detach_()
            view2_target_projection.detach_()
        
        
        

        return (view1_online_prediction,
                view2_target_projection.detach_(),
                view2_online_prediction,
                view1_target_projection.detach_())
                
        
        


class ByolModel(pl.LightningModule):
    def __init__(self,
                backbone: nn.Module,
                criterion: nn.Module = NormalizedMSELoss,
                target_decay: float = 0.996,
                optimizer: str = 'sgd',
                learning_rate: float =  1e-2,
                weight_decay: float = 0.0,
                scheduler: str = 'step',
                sched_step_size: int = 5,
                shced_gamm: float = 0.5
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
        self.sched_gamma = shced_gamm

    def training_step(self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        view1, view2 = batch
        v1_on, v2_tar, v2_on, v1_tar = self.learner(view1,view2)
        loss = self.criterion(v1_on, v2_tar, v2_on, v1_tar)
        
        # logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        return loss

    def on_before_zero_grad(self,_) -> None:
        self.learner.update_target_network()


    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.learner.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )
        elif self.optimizer == 'sgd':
            optimizer = optim.Adam(params=self.learner.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )
        else:
            raise NameError('optimizer must be eithr sgd or adam')


        if self.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                  step_size=self.sched_step_size,
                                                  gamma=self.sched_gamma
                                                 )
        elif self.scheduler == 'exponential':
             scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                          gamma=self.sched_gamma
                                                         )  

        else: 
            raise NameError('scheduler must be eithr step or exponential')
        
        return{'optimizer': optimizer,
               'schedular': scheduler
              }


        






    

    

