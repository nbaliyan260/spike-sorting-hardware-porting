import warnings
import numpy as np
import torch
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from torchbci.kilosort4.parameters import DEFAULT_SETTINGS

logger = logging.getLogger("kilosort")

KS3_SETTINGS_DEFAULTS = {
    "algorithm": "ks3",
    "ks3_min_cluster_size": 200,
    "ks3_n_projections": 25,
    "ks3_score_threshold": 0.35,
    "ks3_max_splits": 512,
    "ks3_random_seed": 0,
    "ks3_use_ccg": True,
    "ks3_ccg_refrac_s": 0.0015,
    "ks3_ccg_window_s": 0.008,
    "ks3_ccg_dip_ratio_thresh": 0.2,
}

RECOGNIZED_SETTINGS = list(DEFAULT_SETTINGS.keys())
RECOGNIZED_SETTINGS.extend([
    'filename', 'data_dir', 'results_dir', 'probe_name', 'probe_path',
])


class InitializeOpsBlock(nn.Module):
    """
    Wraps kilosort.run_kilosort.initialize_ops.
    Builds the ops dictionary from settings+probe and attaches filename if provided.
    """
    def __init__(
        self,
        settings: Dict[str, Any],
        probe: Optional[Dict[str, Any]] = None,
        data_dtype: Optional[Any] = None,
        do_CAR: bool = True,
        invert_sign: bool = False,
        save_preprocessed_copy: bool = False,
        device: Optional[torch.device] = None,
        kilosort_version: int = 4
    ):
        super().__init__()
        self.settings = dict(settings)  # will be modified by initialize_ops
        self.probe = probe
        self.data_dtype = data_dtype
        self.do_CAR = do_CAR
        self.invert_sign = invert_sign
        self.save_preprocessed_copy = save_preprocessed_copy
        self.device = device
        self.kilosort_version = kilosort_version

    def initialize_ops(self, settings, probe, data_dtype, do_CAR, invert_sign,
                   device, save_preprocessed_copy, gui_mode=False) -> dict:
        """Package settings and probe information into a single `ops` dictionary."""

        settings = settings.copy()
        if self.kilosort_version == 3:
            RECOGNIZED_SETTINGS.extend([*KS3_SETTINGS_DEFAULTS.keys()])
            for key, default_value in KS3_SETTINGS_DEFAULTS.items():
                settings.setdefault(key, default_value)

        if settings['nt0min'] is None:
            settings['nt0min'] = int(20 * settings['nt']/61)
        if settings['max_channel_distance'] is None:
            # Default used to be None, now it's a constant. Adding this so that
            # cached settings values in the GUI don't cause disruption.
            settings['max_channel_distance'] = DEFAULT_SETTINGS['max_channel_distance']

        if settings['nearest_chans'] > len(probe['chanMap']):
            msg = f"""
                Parameter `nearest_chans` must be less than or equal to the number 
                of data channels being sorted.\n
                Changing from {settings['nearest_chans']} to {len(probe['chanMap'])}.
                """
            warnings.warn(msg, UserWarning)
            settings['nearest_chans'] = len(probe['chanMap'])

        if 'duplicate_spike_bins' in settings:
            msg = """
                The `duplicate_spike_bins` parameter has been replaced with 
                `duplicate_spike_ms`. Specifying the former will have no effect, 
                since it gets overwritten based on sampling rate.
                """
            warnings.warn(msg, DeprecationWarning)
        dup_bins = int(settings['duplicate_spike_ms'] * (settings['fs']/1000))

        # If running through GUI, also allow some additional relevant keys in
        # settings dictionary.
        recognized = RECOGNIZED_SETTINGS.copy()

        # Raise an error if there are unrecognized settings entries to make users
        # aware if they've made a typo, are using a deprecated setting, etc.
        unrecognized = []
        for k, _ in settings.items():
            if k not in recognized:
                unrecognized.append(k)
        if len(unrecognized) > 0:
            logger.info('Unrecognized keys found in `settings`')
            logger.info('See `kilosort.run_kilosort.RECOGNIZED_SETTINGS`')
            raise ValueError(f'Unrecognized settings: {unrecognized}')


        ops = settings.copy()
        ops['settings'] = settings
        ops['probe'] = probe
        ops['data_dtype'] = data_dtype
        ops['do_CAR'] = do_CAR
        ops['invert_sign'] = invert_sign
        ops['NTbuff'] = ops['batch_size'] + 2 * ops['nt']
        ops['Nchan'] = len(probe['chanMap'])
        ops['n_chan_bin'] = settings['n_chan_bin']
        ops['duplicate_spike_bins'] = dup_bins
        ops['torch_device'] = str(device)
        ops['save_preprocessed_copy'] = save_preprocessed_copy

        if not settings['templates_from_data'] and settings['nt'] != 61:
            raise ValueError('If using pre-computed universal templates '
                            '(templates_from_data=False), nt must be 61')

        ops = {**ops, **probe}

        return ops, settings

    def forward(self) -> Dict[str, Any]:
        # print("Initializing ops...")
        ops, _settings = self.initialize_ops(
            self.settings,
            self.probe,
            self.data_dtype,
            self.do_CAR,
            self.invert_sign,
            self.device,
            self.save_preprocessed_copy,
            gui_mode=False,
        )
                
        return ops
