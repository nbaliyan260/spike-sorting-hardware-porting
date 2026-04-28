import torch
from .block import Block
from typing import List, Tuple

class Alignment(Block):
    """
    Base class for all alignment blocks in torchbci.

    :param Block: _description_
    :type Block: _type_
    """  
    def __init__(self):
        super().__init__()



class JimsAlignment(Alignment):
    """Align detected peaks by removing duplicates within a spatiotemporal neighborhood.

    Args:
        sort_by (str): Criterion to sort peaks before alignment.
            Must be either ``"value"`` (sort by amplitude) or ``"time"`` (sort by index).
        leniency_channel (int, optional): Maximum channel distance to consider peaks duplicates.
            Defaults to 3.
        leniency_time (int, optional): Maximum temporal distance to consider peaks duplicates.
            Defaults to 15.
    """

    def __init__(self, 
                 sort_by: str, 
                 leniency_channel: int = 3, 
                 leniency_time: int = 15, 
                 n_jims_features: int = 5,
                 jims_pad_value: float = 0.0):
        super().__init__()
        self.sort_by = sort_by
        self.leniency_channel = leniency_channel
        self.leniency_time = leniency_time
        self.n_jims_features = n_jims_features
        self.jims_pad_value = jims_pad_value # value to pad jims features with if not enough features are found

    def forward(self, matched_frames, matched_meta) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Align peaks by keeping the most prominent and removing nearby duplicates.

        Args:
            matched_frames (torch.Tensor): Input tensor of shape ``(num_frames, frame_size)``.
            matched_meta (torch.Tensor): Tensor of metadata for each frame.
            vals (torch.Tensor): Amplitudes of detected peaks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]: A tuple containing the aligned frames, their metadata, JIMS features, and full feature vectors.
        """
        peak_vals = matched_frames.max(dim=1).values
        x = torch.concat((matched_meta, peak_vals.unsqueeze(1), matched_frames), dim=1) # (num_frames, 2 + 1 + frame_size)
        if self.sort_by == "value":
            sorted_indices = torch.argsort(x[:, 2], descending=True)
        elif self.sort_by == "time":
            sorted_indices = torch.argsort(x[:, 1])
        else:
            raise ValueError("Invalid sort_by value. Use 'value' or 'time'.")

        aligned_frames = []
        aligned_frames_fullfeatures = []
        aligned_frames_jimsfeatures = []
        aligned_meta = []

        for i in sorted_indices:
            next_frame = False
            chn, time = x[i, 0].item(), x[i, 1].item()
            for u_idx, u in enumerate(aligned_meta):
                chn_diff = chn - u[0]
                if abs(chn_diff) <= self.leniency_channel and abs(time - u[1]) <= self.leniency_time:
                    next_frame = True
                    if chn_diff < 0: # current peak is on a lower channel than the aligned peak
                        aligned_frames_fullfeatures[u_idx].insert(0, x[i, 3:]) # insert before the representative frame
                    elif chn_diff > 0: # current peak is on a higher channel than the aligned peak
                        aligned_frames_fullfeatures[u_idx].append(x[i, 3:]) # append after the representative frame
                    if len(aligned_frames_jimsfeatures[u_idx]) < self.n_jims_features: # only add jims feature if we haven't reached the required number yet
                        if chn_diff < 0:
                            aligned_frames_jimsfeatures[u_idx].insert(0, x[i, 2].item()) # insert before the representative frame
                            aligned_frames_jimsfeatures[u_idx].append(x[i, 2].item()) # add the same value to keep symmetry
                        elif chn_diff > 0:
                            aligned_frames_jimsfeatures[u_idx].append(x[i, 2].item()) # append after the representative frame
                            aligned_frames_jimsfeatures[u_idx].insert(0, x[i, 2].item()) # add the same value to keep symmetry
                    break
            if next_frame:
                continue

            aligned_frames.append(x[i, 3:]) 
            aligned_meta.append([chn, time])
            aligned_frames_fullfeatures.append([x[i, 3:]]) 
            aligned_frames_jimsfeatures.append([x[i, 2].item()])
        
        # append jim feature vectors to the required length
        for j_idx, j in enumerate(aligned_frames_jimsfeatures):
            while len(j) < self.n_jims_features:
                j.insert(0, self.jims_pad_value) # pad with the specified value at the beginning
                j.append(self.jims_pad_value) # pad with the specified value at the end
        aligned_frames_jimsfeatures = torch.tensor(aligned_frames_jimsfeatures)
        aligned_frames = torch.stack(aligned_frames)
        aligned_meta = torch.tensor(aligned_meta)

        # for each in aligned_frames_features, first item is the representative frame, rest are aligned frames to it.
        return aligned_frames, aligned_meta, aligned_frames_jimsfeatures, aligned_frames_fullfeatures