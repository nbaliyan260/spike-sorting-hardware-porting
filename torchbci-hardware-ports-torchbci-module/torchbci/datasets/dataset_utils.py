import os
import numpy as np


def slice_data_and_spikes(
    data_dir,
    st_dir,
    data_sf,
    st_sf,
    n_channels,
    raw_dtype,
    percentage_desired,
    sliced_data_dir,
    sliced_st_dir,
    margin_samples=100       # default margin after final spike
):
    """
    Slice a raw Neuropixels .bin file and corresponding spike-times .npy file.
    Extracts a percentage chunk of the dataset while ensuring a margin after 
    the last spike. Saves both sliced data and spikes.
    """

    # -------------------------
    # Resolve paths
    # -------------------------
    raw_path = data_dir
    st_path = st_dir

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Spike-times file not found: {st_path}")

    # -------------------------
    # Load raw (memmap)
    # -------------------------
    raw = np.memmap(raw_path, dtype=raw_dtype, mode="r")

    if raw.size % n_channels != 0:
        raise ValueError("Raw file size is not divisible by n_channels")

    n_samples = raw.size // n_channels

    # -------------------------
    # Load spike times
    # -------------------------
    spikes_original = np.load(st_path)

    if spikes_original.ndim != 1:
        raise ValueError(
            f"Spike-times array must be 1D, got shape {spikes_original.shape}"
        )

    # -------------------------
    # Convert ST indices to sample units of raw data
    # -------------------------
    conversion_factor = data_sf / st_sf
    spikes_30k = np.round(spikes_original * conversion_factor).astype(np.int64)

    # Keep only valid spike indices
    valid_mask = (spikes_30k >= 0) & (spikes_30k < n_samples)
    spikes_30k = spikes_30k[valid_mask]

    spikes_30k = np.sort(spikes_30k)

    # -------------------------
    # Compute cutoff with margin
    # -------------------------
    if percentage_desired <= 0 or percentage_desired > 1:
        raise ValueError("percentage_desired should be more than 0 and less than 1")
    
    naive_cutoff = int(n_samples * percentage_desired)
    spikes_in_window = spikes_30k[spikes_30k < naive_cutoff]

    if spikes_in_window.size > 0:
        last_spike = int(spikes_in_window.max())
        cutoff = min(last_spike + margin_samples, n_samples)
    else:
        cutoff = naive_cutoff

    # -------------------------
    # Save sliced raw data
    # -------------------------
    os.makedirs(sliced_data_dir, exist_ok=True)
    sliced_data_path = os.path.join(
        sliced_data_dir,
        f"raw_{int(percentage_desired*100)}pct.bin"
    )

    num_values = cutoff * n_channels
    subset_flat = raw[:num_values]
    subset_flat.tofile(sliced_data_path)

    # -------------------------
    # Save sliced spike times
    # -------------------------
    os.makedirs(sliced_st_dir, exist_ok=True)
    sliced_st_path = os.path.join(
        sliced_st_dir,
        f"spikes_{int(percentage_desired*100)}pct.npy"
    )

    subset_spikes = spikes_30k[spikes_30k < cutoff]
    np.save(sliced_st_path, subset_spikes)

    # -------------------------
    # Final message
    # -------------------------
    print("Data and ST are saved")



base_dir = rf"D:\Marquees-smith\c46"
data_file_name = "c46_npx_raw.bin"
st_file_name = "c46_extracellular_spikes.npy"
data_dir = os.path.join(base_dir, data_file_name)
st_dir = os.path.join(base_dir, st_file_name)
sliced_data_dir = os.path.join(base_dir, "subset_data")
sliced_st_dir = os.path.join(base_dir, "subset_st")

extracellular_sampling_frequency = 30000.0
intracellular_sampling_frequency = 50023.87552924, # WARNING: this is specific to c46, if you change to another data such as c14, you will need to use another value
num_channels = 384
raw_dtype = "int16"
percentage_desired = 0.01 # 1% of the data


slice_data_and_spikes(
    data_dir      = data_dir,
    st_dir        = st_dir,
    data_sf       = extracellular_sampling_frequency,
    st_sf         = intracellular_sampling_frequency,
    n_channels    = num_channels,
    raw_dtype     = raw_dtype,
    percentage_desired = percentage_desired,  # 1%
    sliced_data_dir = sliced_data_dir,
    sliced_st_dir   = sliced_st_dir,
)

