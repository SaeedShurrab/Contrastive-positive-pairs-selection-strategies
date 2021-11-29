
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