import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

class NegativeCosineSimilarity(nn.Module):
    def __init__(self,
                 mode: str = 'simplified'
                ) -> None:
        super(NegativeCosineSimilarity,self).__init__()
        
        self.mode = mode
        assert self.mode in ['simplified', 'original'], \
        'loss mode must be either (simplified) or (original)'
        
        
    def _forward1(self,
                  p: Tensor,
                  z: Tensor,
                 ) -> Tensor:
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        loss = -(p*z).sum(dim=1).mean()
        return loss
        
    def _forward2(self,
                  p: Tensor,
                  z: Tensor,
                 ) -> Tensor:
        z = z.detach
        loss = - F.cosine_similarity(p, z, dim=-1).mean()
        return loss
        
    def forward(self,
                  p1: Tensor,
                  p2: Tensor,
                  z1: Tensor,
                  z2: Tensor,
                 ) -> Tensor:
        
        if self.mode == 'original':
            loss1 = self._forward1(p1,z2)
            loss2 = self._forward1(p2,z1)
            loss = loss1/2 +loss2/2
            return loss
        
        elif self.mode == 'simplified':
            loss1 = self._forward1(p1,z2)
            loss2 = self._forward1(p2,z1)
            loss = loss1/2 +loss2/2
            return loss



class ProjectionMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 2048,
                 output_dim: int = 2048,
                ) -> None:
        super(ProjectionMLP,self).__init__()
        
        

        self.layer1 = nn.Sequential(nn.Linear(in_features=input_dim, out_features= hidden_dim, bias=False ),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )

        self.layer2 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )
        
        self.layer3 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim)
                                   )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x




class PredictionMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 512,
                 output_dim: int = 2048,
                ) -> None:
        super(PredictionMLP,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias= False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )
        
        self.layer2 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=output_dim))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
        


class EncodProject(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 hidden_dim: int = 2048,
                 output_dim: int = 2048
                 ) -> None:
        super(EncodProject, self).__init__()
                
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        
        self.projector = ProjectionMLP(input_dim=nn.Sequential(*list(model.children()))[-1].in_features,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim
                                       )
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x



class SimSiam(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 projector_hidden_dim: int = 2048,
                 projector_output_dim: int = 2048,
                 predictor_hidden_dim: int = 512,
                 predictor_output_dim: int = 2048
                ) -> None: 
        super(SimSiam, self).__init__()
        
        self.encode_project = EncodProject(model, 
                                           hidden_dim= projector_hidden_dim,
                                           output_dim= projector_hidden_dim
                                          )
        self.predictor = PredictionMLP(input_dim=projector_output_dim,
                                       hidden_dim=predictor_hidden_dim,
                                       output_dim=predictor_output_dim)
        
    def forward(self, 
                x1: Tensor,
                x2: Tensor
               ) -> Tuple[Tensor]:
        
        f, h = self.encode_project, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        
        
        return {'p1': p1,
                'p2' : p2,
                'z1' : z1,
                'z2' : z2}