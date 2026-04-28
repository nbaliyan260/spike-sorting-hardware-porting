"""Functional Interface"""
import torch


def get_channel_relative_distances(channel_coords):
    """Compute relative distances between channels based on their coordinates.

    Args:
        channel_coords (torch.Tensor): Tensor of shape (num_channels, 2) representing
            the (x, y) coordinates of each channel.

    Returns:
        torch.Tensor: A tensor of shape (num_channels, num_channels, 1) containing the relative
            distances between each pair of channels.
    """
    channel_coords = channel_coords.unsqueeze(0)
    distances = torch.norm(channel_coords - channel_coords.transpose(0, 1), dim=-1, keepdim=True)
    return distances

def delay_and_decay(single_channel_signal, 
                    relative_channel_distance,
                    delay_factor,
                    decay_factor,
                    num_channels_to_keep: int,
                    max_length_to_keep: int,
                    ):
    # delay: shift right by delay_steps
    # decay: reduce amplitude by decay_factor^distance
    num_channels = relative_channel_distance.shape[0]
    delay_steps = (relative_channel_distance * delay_factor).int()
    decay_amplitudes = decay_factor ** relative_channel_distance
    new_length = single_channel_signal.shape[-1] + delay_steps.max().item()
    new_signal = torch.zeros((num_channels, new_length), device=single_channel_signal.device)
    for ch in range(num_channels):
        new_signal[ch, delay_steps[ch]:delay_steps[ch]+single_channel_signal.shape[-1]] = single_channel_signal * decay_amplitudes[ch]
    channels_to_keep = torch.argsort(relative_channel_distance[:, 0])[:num_channels_to_keep]
    new_signal = new_signal[channels_to_keep, :max_length_to_keep]
    return new_signal, channels_to_keep