from torchbci.block import Filter, Detection, Alignment, Block, TemplateMatching
import torch
import numpy as np
from sklearn.decomposition import PCA

# Optional imports — not available in all environments (e.g. Tenstorrent TT-Metal venv)
try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

try:
    from scipy import signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

import torch.nn.functional as F
from typing import List, Tuple

from torchbci.block.clustering import SimpleOnlineKMeansClustering
from torchbci.block.functional import delay_and_decay


class Kilosort4CAR(Filter):
    # common average referencing
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        mean_across_channels = torch.mean(data, dim=0, keepdim=True)
        referenced = data - mean_across_channels
        return referenced

class Kilosort4Filtering(Filter):
    # 300 Hz high-pass butterworth filter
    def __init__(self, sample_rate: int, cutoff_freq):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq

    # torch version has some issues
    # def forward(self, data: torch.Tensor) -> torch.Tensor:
    #     filtered = torchaudio.functional.highpass_biquad(
    #         data, sample_rate=self.sample_rate, cutoff_freq=self.cutoff_freq
    #     )
    #     return filtered
    
    # scipy
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        sos = signal.butter(4, self.cutoff_freq, btype='highpass', fs=self.sample_rate, output='sos')
        filtered = signal.sosfiltfilt(sos, data.numpy(), axis=1).copy()
        return torch.tensor(filtered, device=data.device, dtype=data.dtype)

class Kilosort4Whitening(Filter):
    def __init__(self, 
                 channel_relative_distances: torch.Tensor,
                 nearest_channels: int = 32):
        super().__init__()
        self.channel_relative_distances = channel_relative_distances
        self.nearest_channels = nearest_channels

    @staticmethod
    def ZCA_transform(X: torch.Tensor):
        C = torch.cov(X) # X: (num_channels, num_samples), C: (num_channels, num_channels)
        U, S, V = torch.linalg.svd(C) # U: (num_channels, num_channels), S: (num_channels,), V: (num_channels, num_channels)
        W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-5)) @ U.t() # W: (num_channels, num_channels)
        return W
    
    def forward(self, data: torch.Tensor):
        n_channels = data.shape[0]
        data_whitened = torch.zeros_like(data)
        for channel in range(n_channels):
            channel_data = data[channel].unsqueeze(0) # channel_data: (1, num_samples)
            # get N closest channels based on relative distances
            nearest_channels_id = torch.argsort(self.channel_relative_distances[channel, :, 0])[:self.nearest_channels]
            nearest_channel_data = data[nearest_channels_id]
            # compute ZCA transform on the nearest channels
            W_data = self.ZCA_transform(nearest_channel_data) # W_data: (num_channels, num_channels)
            # apply the whitening transform, but just for the current channel
            channel_whitened = (W_data @ nearest_channel_data)[0] # channel_whitened: (num_samples,)
            data_whitened[channel] = channel_whitened
        return data_whitened
    
class Kilosort4Calibrator(Block):
    def __init__(self):
        super().__init__()
        
    def calibrate_factors(self, 
                          calibrate_segment_length: int,
                          max_lookahead_length: int,
                          calibrate_data: torch.Tensor, 
                          channel_relative_distances: torch.Tensor,
                          labels: List[int],
                          center_channels: List[int],):
        # calibrate delay factors
        best_lookaheads_1 = []
        best_lookaheads_2 = []
        for label, center_channel in zip(labels, center_channels):
            center_segment = calibrate_data[center_channel, label:label+calibrate_segment_length]
            closest_two_channels = channel_relative_distances[center_channel].argsort()[:3]
            similarities_1 = []
            similarities_2 = []
            # we just find how many steps achieve the best similarity (so that we know how much delay to add)
            # e.g. if two channels are 10 um apart, and the best lookahead is 5 steps
            # then delay_factor = 5 steps / 10 um = 0.5 steps/um
            # this information can be used when we build template across channels
            # delay is averaged over two closest channels and also all the labels to be more robust
            for lookahead in range(max_lookahead_length):
                lookahead_data_1 = calibrate_data[closest_two_channels[0], label+lookahead:label+lookahead+calibrate_segment_length]
                lookahead_data_2 = calibrate_data[closest_two_channels[1], label+lookahead:label+lookahead+calibrate_segment_length]
                similarity = F.cosine_similarity(center_segment, lookahead_data_1)
                similarities_1.append(similarity)
                similarity = F.cosine_similarity(center_segment, lookahead_data_2)
                similarities_2.append(similarity)
            best_lookaheads_1.append(torch.argmax(torch.tensor(similarities_1)))
            best_lookaheads_2.append(torch.argmax(torch.tensor(similarities_2)))
        delay_factor_1 = torch.mean(torch.tensor(best_lookaheads_1).float() / channel_relative_distances[center_channels, closest_two_channels[0], 0])
        delay_factor_2 = torch.mean(torch.tensor(best_lookaheads_2).float() / channel_relative_distances[center_channels, closest_two_channels[1], 0])
        delay_factor = (delay_factor_1 + delay_factor_2) / 2.

        for label, center_channel in zip(labels, center_channels):
            center_segment = calibrate_data[center_channel, label:label+calibrate_segment_length]
            closest_two_channels = channel_relative_distances[center_channel].argsort()[:3]
            # when spikes propagate to nearby channels, their amplitude decays
            # we can find the maximum amplitude on the nearby channels and compare with the center channel
            spike_amplitude = torch.max(center_segment) # make sure the spike is positive as some datasets have negative spikes
            max_amplitude_1 = torch.max(calibrate_data[closest_two_channels[0], label:label+max_lookahead_length+calibrate_segment_length])
            max_amplitude_2 = torch.max(calibrate_data[closest_two_channels[1], label:label+max_lookahead_length+calibrate_segment_length])
            # we assume the decay follows an exponential model: amplitude = A * decay_factor^distance
            # so decay_factor = (amplitude / A)^(1/distance)
            decay_factor_1 = (max_amplitude_1 / spike_amplitude) ** (1 / channel_relative_distances[center_channel, closest_two_channels[0], 0])
            decay_factor_2 = (max_amplitude_2 / spike_amplitude) ** (1 / channel_relative_distances[center_channel, closest_two_channels[1], 0])
        decay_factor = (decay_factor_1 + decay_factor_2) / 2.
        return delay_factor.item(), decay_factor.item()

class Kilosort4Detection(Detection):
    def __init__(self, 
                 channel_relative_distances: torch.Tensor,
                 threshold: float,
                 lookbehind_length: int,
                 lookahead_length: int,
                 predefined_templates: torch.Tensor,
                 spike_template_matching_threshold_similarity: float,
                 num_feature_channels: int,
                 max_spikes: int,
                 feature_length: int,
                 delay_factor: float = 0.25,
                 decay_factor: float = 0.96):
        super().__init__()
        self.threshold = threshold
        self.lookbehind_length = lookbehind_length
        self.lookahead_length = lookahead_length
        self.predefined_templates = predefined_templates
        self.spike_template_matching_threshold_similarity = spike_template_matching_threshold_similarity
        self.num_feature_channels = num_feature_channels
        self.max_spikes = max_spikes
        self.feature_length = feature_length

        self.delay_factor = delay_factor
        self.decay_factor = decay_factor

        self.channel_relative_distances = channel_relative_distances

    # detect spikes using predefined templates
    # extract the waveform of the spike of the channel
    # expand the waveform to nearby channels using delay and decay
    # subtract the expanded waveform from the data to remove the spike's impact
    # detect spikes on the residual data again and repeat

    def thresholding(self, data: torch.Tensor):
        data_thresholding = data.clone()
        data_thresholding[data_thresholding < self.threshold] = 0.0
        return data_thresholding, data

    # def template_matching_for_detection(self, 
    #                                     predefined_templates: torch.Tensor,
    #                                     data: torch.Tensor,
    #                                     acceptable_similarity: float):
    #     # predefined_templates: (num_templates, frame_length) as predefined templates are single channel templates
    #     # data: (num_channels, num_samples)
    #     # stop and return when first spike is found (by any of the templates)
    #     # so that we can remove it and then repeat the template matching to detect more spikes
    #     data_length = data.shape[1]
    #     template_kernel_expanded = predefined_templates.unsqueeze(1) # (num_templates, 1, frame_length)
    #     for data_step in range(data_length - predefined_templates.shape[1]):
    #         data_segment = data[:, data_step:data_step + predefined_templates.shape[1]]
    #         # if torch.max(data_segment) < self.threshold:
    #         #     continue # no spikes reach the threshold in this segment
    #         # data_segment: (num_channels, frame_length)
    #         # kernel: (num_templates, num_channels, frame_length)
    #         similarities = F.cosine_similarity(data_segment.unsqueeze(0), template_kernel_expanded, dim=-1) # (num_templates, num_channels)
    #         if torch.max(similarities) >= acceptable_similarity:
    #             max_sim_index = torch.argmax(similarities).item()
    #             template_id = max_sim_index // data.shape[0]
    #             channel_id = max_sim_index % data.shape[0]
    #             return (channel_id, data_step), similarities[template_id, channel_id]
    #     return None, None

    def template_matching_for_detection(
            self,
            predefined_templates: torch.Tensor,  # (T, F)
            data: torch.Tensor,                  # (C, N)
            acceptable_similarity: float,
            eps: float = 1e-12
    ):
        T, F_len = predefined_templates.shape
        C, N = data.shape
        if N < F_len:
            return (None, None)

        device = data.device
        dtype = data.dtype

        # Precompute template norms: (T,)
        template_norms = torch.linalg.vector_norm(predefined_templates, dim=1).clamp_min(eps)

        # Conv kernels
        # Numerator kernels: (T, 1, F_len) — conv1d uses cross-correlation (no flip), which matches dot products.
        num_kernels = predefined_templates.to(device=device, dtype=dtype).unsqueeze(1)  # (T,1,F)

        # Denominator kernel: sliding sum of squares via conv with ones, then sqrt
        ones_kernel = torch.ones((1, 1, F_len), device=device, dtype=dtype)

        best_sim = None
        best_c = None
        best_s = None
        # (Optional) track best template index if you want it later
        # best_t = None

        # Stream over channels to keep memory usage low
        for c in range(C):
            x = data[c:c+1, :].unsqueeze(1)          # (1,1,N)

            # Numerator: dot products with each template at every step -> (1,T,S)
            num = F.conv1d(x, num_kernels, bias=None, stride=1)  # (1,T,S)
            S = num.shape[-1]

            # Segment norms: sqrt(sum(x^2)) for each window -> (1,1,S)
            seg_norm = F.conv1d(x * x, ones_kernel, bias=None, stride=1).clamp_min(eps).sqrt()  # (1,1,S)

            # Denominator per (template, step): (1,T,S)
            denom = template_norms.view(1, T, 1) * seg_norm  # broadcast

            # Cosine similarity
            sims = num / (denom + eps)  # (1,T,S)

            # Find local max for this channel
            local_max, local_idx = sims.view(-1).max(dim=0)
            if best_sim is None or local_max > best_sim:
                best_sim = local_max
                # unravel local_idx into (t, s)
                t_idx = (local_idx // S).item()
                s_idx = (local_idx % S).item()
                best_c = c
                best_s = s_idx
                # best_t = t_idx

        if best_sim is not None and best_sim.item() >= acceptable_similarity:
            # Return (channel_id, data_step), similarity tensor
            return (best_c, best_s), best_sim
        else:
            return (None, None)
           
    # def spike_extraction(self, data: torch.Tensor, spikes_indices: Tuple[torch.Tensor, torch.Tensor]):
    #     # frames: lookbehind_length + 1 + lookahead_length
    #     spikes_frame_indices = torch.zeros((len(spikes_indices[0]), self.lookbehind_length + 1 + self.lookahead_length))
    #     for i in range(len(spikes_indices[0])):
    #         channel = spikes_indices[0][i]
    #         time = spikes_indices[1][i]
    #         spikes_frame_indices[i, 0] = channel
    #         spikes_frame_indices[i, 1:] = torch.arange(time - self.lookbehind_length, time + self.lookahead_length + 1)
    #     spike_frames = data[spikes_frame_indices[:, 0].long(), spikes_frame_indices[:, 1:].long()] # (num_spikes, frame_length)
    #     return spikes_indices.long(), spike_frames

    def spike_extraction(self, data: torch.Tensor, spikes_indice: Tuple[int, int]):
        # just one spike
        return data[spikes_indice[0], spikes_indice[1]-self.lookbehind_length:spikes_indice[1]+self.lookahead_length+1]
    

    def iterative_spike_detection(self, 
                                  data: torch.Tensor,):
        detected_spikes = []
        detected_similarities = []
        spike_features = []
        residual_data = data.clone()
        for _ in range(self.max_spikes):
            spike_indice, similarity = self.template_matching_for_detection(
                self.predefined_templates, residual_data, self.spike_template_matching_threshold_similarity
            )
            print(spike_indice, similarity)
            if spike_indice is None:
                break
            detected_spikes.append([spike_indice[0], spike_indice[1]])
            detected_similarities.append(similarity)
            # extract the spike waveform
            spike_waveform = self.spike_extraction(residual_data, spike_indice) # (1, frame_length)
            # expand the spike waveform to nearby channels using delay and decay
            channel_id = spike_indice[0]
            relative_channel_distances = self.channel_relative_distances[channel_id] # (num_channels, 1)
            # option 1: propagate by the delay and decay factors
            expanded_waveform, expanded_channels = delay_and_decay(
                spike_waveform, 
                relative_channel_distances,
                self.delay_factor,
                self.decay_factor,
                self.num_feature_channels,
                self.feature_length, # feature_length
            ) # (num_channels, new_length)
            start_index = spike_indice[1] - self.lookbehind_length
            end_index = start_index + expanded_waveform.shape[1]
            if end_index > residual_data.shape[1]:
                end_index = residual_data.shape[1]
                expanded_waveform = expanded_waveform[:, :end_index - start_index]
            residual_data[expanded_channels, start_index:end_index] -= expanded_waveform


            # option 2: take the nearby channels from the residue data directly
            # expanded_channels = torch.argsort(relative_channel_distances[:, 0])[:self.num_feature_channels]
            # expanded_waveform = torch.zeros((self.num_feature_channels, spike_waveform.shape[0]), device=data.device)
            # expanded_waveform = residual_data[expanded_channels, spike_indice[1]-self.lookbehind_length:spike_indice[1]+self.lookahead_length+1]
            # start_index = spike_indice[1] - self.lookbehind_length
            # end_index = start_index + expanded_waveform.shape[1]
            # residual_data[expanded_channels, start_index:end_index] -= expanded_waveform

            
            # spike_features.append(expanded_waveform)
            spike_features.append(expanded_waveform[0]) # just take the main channel feature
        detected_spikes = torch.tensor(detected_spikes, device=data.device)
        if spike_features:
            spike_features = torch.stack(spike_features)
        else:
            spike_features = torch.empty((0, self.feature_length), device=data.device)
        return detected_spikes, spike_features
    
    forward = iterative_spike_detection

# clustering cant be real time
# class Kilosort4Clustering(Block):
#     def __init__(self, features):
#         super().__init__()
#         self.features = features

#     def find_neighbors(self):
#         pass

#     def reassign_neighbors(self):
#         pass

#     def merging_tree(self):
#         pass
class Kilosort4PCFeatureConversion(Block):
    """
    PCA-based feature compressor.
    Expects X with shape [n_samples, n_features]. If you pass more dims,
    it flattens the trailing dims to features and restores on output.
    """
    def __init__(self, dim_pc_features: int, *, center: bool = True,
                 use_lowrank: bool = True, whiten: bool = False, eps: float = 1e-8):
        super().__init__()
        self.dim_pc_features = int(dim_pc_features)
        self.center = bool(center)
        self.use_lowrank = bool(use_lowrank)
        self.whiten = bool(whiten)
        self.eps = float(eps)

        # Buffers track learned PCA state. Start empty; filled after .fit()
        self.register_buffer("mu_", torch.empty(0))          # [1, D]
        self.register_buffer("components_", torch.empty(0))  # [D, K]
        self.register_buffer("singular_values_", torch.empty(0))  # [K]
        self.register_buffer("explained_variance_", torch.empty(0))       # [K]
        self.register_buffer("explained_variance_ratio_", torch.empty(0)) # [K]
        self.fitted_ = False

    # ---- utils ----
    def _flatten_features(self, X: torch.Tensor):
        if X.dim() == 2:
            orig_shape = None
            return X, orig_shape
        # Collapse trailing dims to features, keep batch/sample dim
        n = X.size(0)
        f = int(torch.tensor(X.shape[1:]).prod().item())
        Xf = X.reshape(n, f)
        return Xf, X.shape

    def _unflatten_features(self, Z: torch.Tensor, orig_shape):
        # For compressed codes we don't restore feature shape (they're K-dim),
        # so only used for reconstructions to original feature space.
        if orig_shape is None:
            return Z
        n = orig_shape[0]
        return Z.reshape(n, *orig_shape[1:])

    def _check_fitted(self):
        if not self.fitted_ or self.components_.numel() == 0:
            raise RuntimeError("PCA not fitted yet. Call .fit(X) or .fit_transform(X) first.")

    # ---- core PCA math ----
    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """
        Learn PCA basis from X: [n_samples, n_features] (or more dims; will flatten).
        Stores mean, components, singular values, explained variance, etc.
        """
        Xf, _ = self._flatten_features(X)
        assert Xf.dim() == 2, "X must be >=2D with samples on dim 0."

        N, D = Xf.shape
        K = min(self.dim_pc_features, D, N)  # safety clamp

        mu = Xf.mean(dim=0, keepdim=True) if self.center else torch.zeros(1, D, device=Xf.device, dtype=Xf.dtype)
        Xc = Xf - mu if self.center else Xf

        if self.use_lowrank:
            # q must be <= min(N, D)
            U, S, V = torch.pca_lowrank(Xc, q=K)  # V: [D, K]
            components = V[:, :K]
            S = S[:K]
        else:
            # Full SVD; more exact, heavier
            U, S, Vh = torch.linalg.svd(Xc, full_m=False)  # Vh: [K', D]
            components = Vh.transpose(0, 1)[:, :K]         # [D, K]
            S = S[:K]

        # Explained variance (eigenvalues of covariance)
        # cov = (Xc^T Xc) / (N - 1); eigvals = S^2 / (N - 1)
        denom = max(N - 1, 1)
        explained_variance = (S**2) / denom
        total_var = (Xc.pow(2).sum(dim=0).sum() / denom) if N > 1 else explained_variance.sum()
        explained_variance_ratio = explained_variance / (total_var + self.eps)

        # Save
        self.mu_ = mu
        self.components_ = components
        self.singular_values_ = S
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.fitted_ = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project to K dims. Returns Z: [n_samples, K]
        """
        self._check_fitted()
        Xf, _ = self._flatten_features(X)
        Xc = Xf - self.mu_ if self.center else Xf
        Z = Xc @ self.components_  # [N, K]
        if self.whiten:
            Z = Z / (self.singular_values_ + self.eps)
        return Z

    @torch.no_grad()
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: torch.Tensor, *, restore_shape: bool = False, orig_shape=None) -> torch.Tensor:
        """
        Reconstruct to original feature space (approx).
        If restore_shape=True and orig_shape is provided, reshape back.
        """
        self._check_fitted()
        Zr = Z * (self.singular_values_ + self.eps) if self.whiten else Z
        Xr = Zr @ self.components_.T
        Xr = Xr + self.mu_ if self.center else Xr
        if restore_shape:
            if orig_shape is None:
                raise ValueError("orig_shape must be provided to restore the original shape.")
            Xr = self._unflatten_features(Xr, orig_shape)
        return Xr

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Alias for transform so you can plug it into a model.
        """
        return self.transform(X)

    # Convenience: % variance explained by the kept components
    @property
    def variance_explained(self) -> float:
        self._check_fitted()
        return float(self.explained_variance_ratio_.sum().item())
    
class Kilosort4Algorithm(Block):
    def __init__(self,
                 sample_rate,
                 channel_relative_distances: torch.Tensor,
                 high_pass_filter_cutoff: int,
                 detection_threshold: float,
                 lookbehind_length: int,
                 lookahead_length: int,
                 num_feature_channels: int,
                 max_spikes: int,
                 feature_length: int,
                 n_clusters: int,
                 predefined_templates: torch.Tensor,
                 spike_template_matching_threshold_similarity: float,
                 dim_pc_features: int,
                 hop_samples: int,
                 margin: int,
                 window_samples: int,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channel_relative_distances = channel_relative_distances
        self.high_pass_filter_cutoff = high_pass_filter_cutoff
        self.detection_threshold = detection_threshold
        self.lookbehind_length = lookbehind_length
        self.lookahead_length = lookahead_length
        self.num_feature_channels = num_feature_channels
        self.max_spikes = max_spikes
        self.feature_length = feature_length
        self.n_clusters = n_clusters
        self.predefined_templates = predefined_templates
        self.spike_template_matching_threshold_similarity = spike_template_matching_threshold_similarity
        self.dim_pc_features = dim_pc_features
        self.H = int(hop_samples)
        self.margin = int(margin)
        self.W = int(window_samples)
        self.CAR = Kilosort4CAR()
        self.filtering = Kilosort4Filtering(sample_rate=self.sample_rate, cutoff_freq=self.high_pass_filter_cutoff)
        self.whitening = Kilosort4Whitening(channel_relative_distances=self.channel_relative_distances)
        # self.calibrator = Kilosort4Calibrator()
        self.detection = Kilosort4Detection(
            channel_relative_distances=self.channel_relative_distances,
            threshold=self.detection_threshold,
            lookbehind_length=self.lookbehind_length,
            lookahead_length=self.lookahead_length,
            predefined_templates=self.predefined_templates,
            spike_template_matching_threshold_similarity=self.spike_template_matching_threshold_similarity,
            num_feature_channels=self.num_feature_channels,
            max_spikes=self.max_spikes,
            feature_length=self.feature_length,
        )
        # PCA: project from feature_length → dim_pc_features
        self.pc_featuring = Kilosort4PCFeatureConversion(dim_pc_features=self.dim_pc_features)
        self._pca_fitted = False  # lazy-fit flag

        # TODO: Device needs to be dynamic
        self.spike_pc_features = torch.empty((0, self.feature_length), device="cpu", dtype=torch.float32)
        self.detected_spikes   = torch.empty((0, 2), device="cpu", dtype=torch.long)
        # Clustering operates on PCA-compressed features (dim_pc_features)
        self.clustering = SimpleOnlineKMeansClustering(
            n_clusters=self.n_clusters,
            cluster_feature_size=self.dim_pc_features,
        )
    def forward_original(self, x: torch.Tensor):
        x = self.CAR(x)
        x = self.filtering(x)
        x = self.whitening(x)
        spikes, spike_features = self.detection(x)
        print(f"spikes shape {spikes.shape}, spike features shape {spike_features.shape}")

        # ── PCA Feature Integration (Nazish, 2026-04-29) ──────────────────────
        # Previously: spike_pc_features = spike_features  (bypass — PCA unused)
        # Now: fit PCA once, then project to dim_pc_features dimensions
        if spike_features is not None and spike_features.shape[0] > 0:
            if not self._pca_fitted:
                self.pc_featuring.fit(spike_features)
                self._pca_fitted = True
            spike_pc_features = self.pc_featuring.transform(spike_features)  # [N, dim_pc_features]
        else:
            spike_pc_features = spike_features
        # ─────────────────────────────────────────────────────────────────────

        if spikes.shape[0] == 0:
            return torch.empty((0, self.dim_pc_features), device=x.device), torch.empty((0, 2), device=x.device), torch.empty((0,), device=x.device)
        clusters, centroids, clusters_meta = self.clustering(spike_pc_features, spikes)
        return clusters, centroids, clusters_meta
    
    def forward(self, x: torch.Tensor, batch_no: int, is_last: bool = False):
        x = self.CAR(x)
        x = self.filtering(x)
        x = self.whitening(x)
        # x, x_original = self.detection.thresholding(x)
        print(f"Running detection on batch number {batch_no} and x shape is {x.shape}")
        #Return local indicies but I am using batches so we need to fix indicies
        spikes, spike_features = self.detection(x)
        
        if spikes.numel():
            core_mask = (spikes[:, 1] >= self.margin) & (spikes[:, 1] < self.margin + self.W)
            spikes = spikes[core_mask]
            spike_features = spike_features[core_mask]

            # Convert local time -> absolute time using the hop
            spikes = spikes.to(dtype=torch.long)  # indices should be long
            spikes[:, 1].add_(batch_no * self.H)

            # Return the number of elements to avoid concatenation errors
            self.detected_spikes = torch.cat([self.detected_spikes, spikes], dim=0)   # (K_total, 2)

        if spike_features.numel():
            self.spike_pc_features = torch.cat([self.spike_pc_features, spike_features], dim=0)

        # only cluster on the last batch
        if not is_last:
            return None, None, None

        # last batch: if nothing detected overall, return empties
        if self.detected_spikes.numel() == 0:
            empty_feats = torch.empty((0, self.dim_pc_features), device=x.device, dtype=x.dtype)
            empty_idx   = torch.empty((0, 2), device=x.device, dtype=torch.long)
            empty_meta  = torch.empty((0, 2), device=x.device, dtype=torch.long)
            return empty_feats, empty_idx, empty_meta

        # ── PCA Feature Integration (Nazish, 2026-04-29) ──────────────────────
        # Fit PCA on all accumulated raw features, then compress before clustering.
        # This is more numerically stable than per-batch fitting.
        # Previously: clustering received raw spike_features (61-dim)
        # Now:        clustering receives PCA-compressed features (dim_pc_features-dim)
        if self.spike_pc_features.shape[0] > 0:
            if not self._pca_fitted:
                self.pc_featuring.fit(self.spike_pc_features)
                self._pca_fitted = True
            compressed_features = self.pc_featuring.transform(self.spike_pc_features)  # [N, dim_pc_features]
        else:
            compressed_features = self.spike_pc_features
        # ─────────────────────────────────────────────────────────────────────

        # cluster on ALL accumulated spikes with PCA-compressed features
        clusters, centroids, clusters_meta = self.clustering(compressed_features, self.detected_spikes)
        return clusters, centroids, clusters_meta

    def run_one_batch(self, batch: torch.Tensor, batch_no:int , total_batches: int):
        """
        Accepts either [C, N] or [B, C, N].
        Returns: (clusters, centroids, clusters_meta) for this batch or list of such tuples for B>1.
        """
        if batch.ndim == 2:         # [C, N]
            print(f"Batch shape is {batch.shape}")
            return self.forward(batch, batch_no, is_last=(batch_no == total_batches - 1))
        elif batch.ndim == 3:       # [B, C, N]
            outs = []
            for b in range(batch.size(0)):
                outs.append(self.forward(batch[b], batch_no ,is_last=(batch_no == total_batches - 1)))
            return outs
        else:
            raise ValueError(f"Expected [C,N] or [B,C,N], got shape {tuple(batch.shape)}")
    def run(self, dataloader):
        """
        Iterate a PyTorch DataLoader that yields [C, N] (or [B, C, N]) windows.
        Yields per-batch results, allowing streaming/online clustering to update internally.
        """
        total_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
            print(f"Running Kilosort4Algorithm on batch with shape {batch.shape} (#{i})")
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.contiguous()

            # TODO: move to device if needed later, for now only CPU
            if batch.is_cuda:
                batch = batch.cpu()
            yield self.run_one_batch(batch, i, total_batches)

    # def cluster(self):
    #     pass
