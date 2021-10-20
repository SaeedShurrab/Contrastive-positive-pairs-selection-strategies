import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


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
    
    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
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
                
        
        
