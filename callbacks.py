import skorch
from skorch.callbacks import Callback
import os
from pathlib import Path


class LoadEndState(Callback):
    def __init__(self, checkpoint: skorch.callbacks.Checkpoint, delete_checkpoint=False):
        """
        Args:
            delele_checkpoints: Deletes checkpoint after loading it.
        """
        self.checkpoint = checkpoint
        self.delete_checkpoint = delete_checkpoint

    def on_train_end(self, net,
                     X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)
        if(self.delete_checkpoint):
            os.remove(Path(self.checkpoint.dirname) / self.checkpoint.f_params)

