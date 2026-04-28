import torch
from .block import Block
from typing import List, Tuple


class TemplateMatching(Block):
    """Base class for all template matching blocks.

    Args:
        Block (torch.nn.Module): Inherits from the Block class.
    """    
    def __init__(self):
        super().__init__()



class JimsTemplateMatching(TemplateMatching):
    """Template matching module for removing outliers based on correlation with reference templates.

    Args:
        templates (torch.Tensor): Tensor of templates to match against.
        similarity_mode (str, optional): Similarity metric to use. 
            Defaults to "cosine".
    """
    def __init__(self, 
                 templates: torch.Tensor, 
                 similarity_mode: str = "cosine",
                 outlier_threshold: float = 0.5):
        super().__init__()
        self.templates = templates
        self.similarity_mode = similarity_mode
        self.outlier_threshold = outlier_threshold

    def forward(self, frames: torch.Tensor, frames_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove outliers that do not match the templates.

        Args:
            frames (torch.Tensor): Input tensor of shape ``(num_frames, frame_size)``.
            frames_meta (torch.Tensor): Tensor of metadata for each frame.

        Raises:
            ValueError: If the similarity mode is invalid.

        Returns:
            Tuple[torch.Tensor, List[Tuple[int, int]]]: A tuple containing the filtered frames and their metadata.
        """        
        sims_all_templates = []
        for template in self.templates:
            if self.similarity_mode == "cosine":
                sims = F.cosine_similarity(template.unsqueeze(0), frames, dim=1) # (num_frames,)
            else:
                raise ValueError("Invalid similarity mode. Use 'cosine'.")
            sims_all_templates.append(sims)
        sims_all_templates = torch.stack(sims_all_templates) # (num_templates, num_frames)
        max_sims, _ = sims_all_templates.max(dim=0) # (num_frames,)
        keep_indices = (max_sims >= self.outlier_threshold).nonzero(as_tuple=True)[0]
        matched_frames = frames[keep_indices]
        matched_meta = frames_meta[keep_indices]
        return matched_frames, matched_meta
