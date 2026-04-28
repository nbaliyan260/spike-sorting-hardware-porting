import spikeinterface as si  # local import
import numpy as np
import logging
import time

from torch import nn
from torchbci.kilosort4.io import load_probe
logger = logging.getLogger("rtsort_pipeline")

class LoadRecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def recording_from_bin_and_probe(
        self,
        bin_path: str,
        probe_path: str,
        sampling_frequency: float,
        num_channels_in_file: int,
        dtype: str = "int16",
        time_axis: int = 0,
    ):
        """
        Returns a SpikeInterface Recording with correct channel order + locations set,
        using a Kilosort chanMap(.mat) / .prb / .json loaded by your load_probe().
        """
        probe = load_probe(probe_path)

        # 1) Load raw binary (this has channels in FILE ORDER 0..num_channels_in_file-1)
        rec = si.read_binary(
            file_paths=bin_path,
            sampling_frequency=sampling_frequency,
            num_channels=num_channels_in_file,
            dtype=dtype,
            time_axis=time_axis,
        )
        # 2) Reorder/select channels to match probe['chanMap'] order (connected-only)
        # probe['chanMap'] are 0-based indices into the original file channels
        ch_ids = rec.get_channel_ids()
        # print(f"Original channel IDs in file order: {ch_ids}")
        sel_ids = ch_ids[probe["chanMap"]]          # reorder + drop disconnected
        rec = rec.select_channels(sel_ids)
        # print(f"Selected channel IDs after reordering to probe chanMap: {rec.get_channel_ids()}")
        # 3) Attach locations as a SpikeInterface property named "location"
        # Shape must be (n_channels, 2) for xy
        locs = np.c_[probe["xc"], probe["yc"]].astype(np.float32)
        rec.set_property("location", locs)

        # sanity checks
        assert rec.get_num_channels() == locs.shape[0], (rec.get_num_channels(), locs.shape)
        _ = rec.get_channel_locations()  # should no longer raise

        return rec

    def forward(self, bin_path, probe_path, sampling_frequency_hz, num_channels_in_file, dtype, time_axis) -> dict:
        tic = time.time()
        rec = self.recording_from_bin_and_probe(
            bin_path=bin_path,
            probe_path=probe_path,
            sampling_frequency=sampling_frequency_hz,
            num_channels_in_file=num_channels_in_file,
            dtype=dtype,
            time_axis=time_axis,
        )
        elapsed = time.time() - tic
        return rec, elapsed
