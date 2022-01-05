from typing import Dict
from torch import Tensor
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class MLFlowLoggerCheckpointer(MLFlowLogger):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, model_checkpoint: ModelCheckpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint.
        """

        self.experiment.log_artifact(
            self.run_id, model_checkpoint.best_model_path
        )



def parse_weights(weights: Dict[str,Tensor]) -> Dict[str,Tensor]:
    
    for k in list(weights.keys()):
        
        if k.startswith('backbone.'):
            
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                
                weights[k[len("backbone."):]] = weights[k]
                
        del weights[k]
        
    return weights