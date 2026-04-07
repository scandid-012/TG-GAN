real_samples_for_gan = scaled_segmented_eeg_data
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math

# Assume real_samples_for_gan is your numpy array of shape (94527, 3, 500)
# and is already scaled to [-1, 1]

# Permute data: (N, C, L) -> (N, L, C)
if isinstance(real_samples_for_gan, np.ndarray): # Ensure it's a numpy array first
    real_samples_for_gan_permuted = np.transpose(real_samples_for_gan, (0, 2, 1))
    print(f"Original data shape: {real_samples_for_gan.shape}")
    print(f"Permuted data shape for Transformer: {real_samples_for_gan_permuted.shape}") # Should be (94527, 500, 3)
else:
    # Handle case where it might already be a tensor or needs conversion
    if torch.is_tensor(real_samples_for_gan):
        if real_samples_for_gan.shape == (94527, 3, 500):
             real_samples_for_gan_permuted = real_samples_for_gan.permute(0, 2, 1)
             print(f"Permuted tensor data shape for Transformer: {real_samples_for_gan_permuted.shape}")
        elif real_samples_for_gan.shape == (94527, 500, 3):
             real_samples_for_gan_permuted = real_samples_for_gan # Already permuted
             print(f"Data shape is already suitable for Transformer: {real_samples_for_gan_permuted.shape}")
        else:
            raise ValueError(f"Unexpected data shape: {real_samples_for_gan.shape}")
    else:
        raise TypeError("real_samples_for_gan should be a NumPy array or PyTorch tensor.")


# Convert to PyTorch Tensor
real_eeg_tensor = torch.tensor(real_samples_for_gan_permuted, dtype=torch.float32)

# Create DataLoader
batch_size = 64 # Choose an appropriate batch size
dataset = TensorDataset(real_eeg_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# EEG segment parameters (will be used by models)
sequence_length = real_eeg_tensor.shape[1] # Should be 500
num_eeg_channels = real_eeg_tensor.shape[2] # Should be 3
