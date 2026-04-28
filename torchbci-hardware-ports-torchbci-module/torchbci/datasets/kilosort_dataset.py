from torch.utils.data import Dataset
import torch

class KilosortDataset(Dataset):
    """
    Splits a continuous multi-channel recording [C, N_total] into overlapping windows [C, N_win].

    Args:
        data: torch.Tensor of shape [C, N_total].
        window_samples: length of each window (N_win).
        hop_samples: stride between consecutive windows.
        margin: extra samples on each side to mitigate boundary effects
                (e.g., lookbehind_length + lookahead_length).
    """
    def __init__(self, data: torch.Tensor, window_samples: int,
                 hop_samples: int, margin: int = 0, transform=None):
        assert data.ndim == 2, "Expected [C, N_total]"
        self.data = data
        self.C, self.N = data.shape
        self.W = window_samples
        self.H = hop_samples
        self.margin = margin

        # Effective window including margins.
        self.W_eff = self.W + 2 * self.margin
        if self.W_eff > self.N:
            raise ValueError("window_samples + 2*margin exceeds total length")

        # Number of windows given hop/stride.
        # Last window starts at idx such that idx + W_eff <= N.
        self.num_windows = 1 + max(0, (self.N - self.W_eff) // self.H)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.H
        end = start + self.W_eff
        x = self.data[:, start:end]  # [C, W_eff]

        # Keep margins so operators with look-ahead/behind work safely
        # and let the downstream block decide what internal indices are valid.
        return x
