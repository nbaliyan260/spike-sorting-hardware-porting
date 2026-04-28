from torchbci.block import Block
from torchbci.block.filter import JimsFilter
from torchbci.block.detection import JimsDetection
from torchbci.block.alignment import JimsAlignment
from torchbci.block.featureselection import JimsFeatureSelection
from torchbci.block.templatematching import JimsTemplateMatching
from torchbci.block.clustering import SimpleOnlineKMeansClustering
import torch
import scipy
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple

class JimsAlgorithm(Block):
    """Complete spike detection pipeline combining filtering, detection, and alignment.
        Args:
            window_size (int, optional): Size of the smoothing filter window. Defaults to 21.
            minimal_impact (int, optional): Minimum cumulative amplitude for a valid peak. Defaults to 5.
            threshold (int, optional): Threshold value for initial detection. Defaults to 50.
            sort_by (str, optional): Sorting method for alignment. Either ``"value"`` or ``"time"``.
                Defaults to ``"value"``.
            leniency_channel (int, optional): Channel tolerance for duplicate suppression. Defaults to 3.
            leniency_time (int, optional): Temporal tolerance for duplicate suppression. Defaults to 15.
            templates (torch.Tensor, optional): Tensor of templates for template matching. Defaults to empty tensor.
            similarity_mode (str, optional): Similarity metric for template matching. Defaults to ``"cosine"``.
    """     
 

    def __init__(self,
                 window_size: int = 21,
                 threshold: int = 50,
                 frame_size: int = 7,
                 normalize: str = "none",
                 sort_by: str = "value",
                 leniency_channel: int = 3,
                 leniency_time: int = 15,
                 templates: torch.Tensor = torch.empty(0),
                 similarity_mode: str = "cosine",
                 outlier_threshold: float = 0.25,
                 n_clusters: int = 64,
                 cluster_feature_size: int = 5,
                 n_jims_features: int = 5,
                 jims_pad_value: float = 0.0):
  
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.frame_size = frame_size
        self.normalize = normalize
        self.sort_by = sort_by
        self.leniency_channel = leniency_channel
        self.leniency_time = leniency_time
        self.templates = templates
        self.similarity_mode = similarity_mode
        self.n_jims_features = n_jims_features
        self.jims_pad_value = jims_pad_value
        self.outlier_threshold = outlier_threshold

        self.filter = JimsFilter(window_size=self.window_size)
        self.detection = JimsDetection(
            threshold=self.threshold
        )
        self.alignment = JimsAlignment(
            sort_by=self.sort_by,
            leniency_channel=self.leniency_channel,
            leniency_time=self.leniency_time,
            n_jims_features=self.n_jims_features,
            jims_pad_value=self.jims_pad_value
        )
        self.template_matching = JimsTemplateMatching(
            templates=self.templates,
            similarity_mode=self.similarity_mode,
            outlier_threshold=self.outlier_threshold
        )
        self.feature_selection = JimsFeatureSelection(
            frame_size=self.frame_size,
            normalize=self.normalize
        )
        self.clustering = SimpleOnlineKMeansClustering(
            n_clusters=n_clusters,
            cluster_feature_size=cluster_feature_size,
        )


    def forward(self, x: torch.Tensor):  #-> Tuple[torch.Tensor, torch.Tensor]
        """Run the full spike detection pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape ``(channels, time)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: frames and metadata after alignment.
        """
        x = self.filter(x)
        peaks, peak_vals = self.detection(x)
        # print(peaks.shape)
        frames, frames_meta = self.feature_selection(x, peaks)
        # print(frames.shape)
        # matched_frames, matched_meta = self.template_matching(frames, frames_meta)
        matched_frames, matched_meta = frames, frames_meta
        # print(matched_frames.shape)
        aligned_frames, aligned_meta, aligned_frames_jimsfeatures, aligned_frames_fullfeatures = self.alignment(matched_frames, matched_meta)
        # print(aligned_frames.shape)
        clusters, centroids, clusters_meta = self.clustering(aligned_frames_jimsfeatures, aligned_meta)
        return clusters, centroids, clusters_meta
    