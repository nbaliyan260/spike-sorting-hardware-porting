import torch
from torch import nn
from typing import Any, Optional
from torchbci.block.base.base_preprocessing_block import BasePreprocessingBlock

class PreprocessingBlock(BasePreprocessingBlock):
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device

    def get_run_parameters(self, ops) -> list:
        """Get `ops` dict values needed by `run_kilosort` subroutines."""
        parameters = [
            ops['settings']['n_chan_bin'],
            ops['settings']['fs'],
            ops['settings']['batch_size'],
            ops['settings']['nt'],
            ops['settings']['nt0min'],  # also called twav_min
            ops['probe']['chanMap'],
            ops['data_dtype'],
            ops['do_CAR'],
            ops['invert_sign'],
            ops['probe']['xc'],
            ops['probe']['yc'],
            ops['settings']['tmin'],
            ops['settings']['tmax'],
            ops['settings']['artifact_threshold'],
            ops['settings']['shift'],
            ops['settings']['scale'],
            ops['settings']['batch_downsampling']
        ]
        return parameters