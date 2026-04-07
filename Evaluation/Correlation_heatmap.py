import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
save_plots=True
output_dir="/kaggle/working/"

def plot_real_and_fake_eeg_heatmaps(real_eeg_data, fake_eeg_data, sfreq, n_seconds=5, channel_names=None, dataset_index=0, sample_index=0):
    """
    Plots real and fake EEG segments as heatmaps side-by-side for comparison.
    
    Parameters:
    - real_eeg_data: numpy array of shape (num_channels, total_samples) for real EEG
    - fake_eeg_data: numpy array of shape (num_samples, sequence_length, num_channels) for fake EEG
    - sfreq: Sampling frequency (Hz)
    - n_seconds: Duration to plot (seconds)
    - channel_names: List of channel names (optional)
    - dataset_index: Index of the real EEG dataset (for title)
    - sample_index: Index of the fake EEG sample to plot
    """
  
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(real_eeg_data.shape[0])]
    
    num_channels = real_eeg_data.shape[0]
    time_samples_to_plot = int(n_seconds * sfreq)
    
    # Ensure we don't exceed available samples
    real_samples = real_eeg_data.shape[1]
    fake_samples = fake_eeg_data.shape[1]
    time_samples_to_plot = min(time_samples_to_plot, real_samples, fake_samples)
    
    # Create time labels for x-axis (in seconds)
    time_ticks = np.arange(0, time_samples_to_plot, int(sfreq/4))  # Tick every 0.25 seconds
    time_labels = [f"{t/sfreq:.2f}" for t in time_ticks]
    
    # Create subplots: left for real EEG, right for fake EEG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Prepare real EEG data for heatmap
    real_eeg_segment = real_eeg_data[:, :time_samples_to_plot]
    
    # Plot real EEG heatmap
    im1 = ax1.imshow(real_eeg_segment, 
                     aspect='auto', 
                     cmap='RdBu_r', 
                     interpolation='bilinear')
    ax1.set_xlabel("Time (samples)")
    ax1.set_ylabel("EEG Channels")
    ax1.set_title(f"Real EEG Heatmap - Dataset {dataset_index} (First {n_seconds}s)")
    ax1.set_yticks(range(num_channels))
    ax1.set_yticklabels(channel_names)
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels)
    
    # Add colorbar for real EEG
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Amplitude')
    
    # Prepare fake EEG data for heatmap (transpose to match real EEG shape)
    fake_eeg_sample = fake_eeg_data[sample_index, :time_samples_to_plot, :].T  # Shape: (num_channels, time_samples)
    
    # Plot fake EEG heatmap
    im2 = ax2.imshow(fake_eeg_sample, 
                     aspect='auto', 
                     cmap='RdBu_r', 
                     interpolation='bilinear')
    ax2.set_xlabel("Time (samples)")
    ax2.set_title(f"Fake EEG Heatmap - Sample {sample_index} (First {n_seconds}s)")
    ax2.set_yticks(range(num_channels))
    ax2.set_yticklabels(channel_names)
    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels(time_labels)
    
    # Add colorbar for fake EEG
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Amplitude')
    
    plt.tight_layout()
    if save_plots:
        plot_path = os.path.join(output_dir, "real-and-fake-eeg-heatmaps.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    plt.show()

def plot_multi_sample_heatmap(fake_eeg_data, sfreq, n_samples=10, n_seconds=2, channel_names=None):
    """
    Plots multiple fake EEG samples as a combined heatmap.
    
    Parameters:
    - fake_eeg_data: numpy array of shape (num_samples, sequence_length, num_channels)
    - sfreq: Sampling frequency (Hz)
    - n_samples: Number of samples to include in heatmap
    - n_seconds: Duration to plot per sample (seconds)
    - channel_names: List of channel names (optional)
    """
    num_channels = fake_eeg_data.shape[2]
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    time_samples_to_plot = int(n_seconds * sfreq)
    n_samples = min(n_samples, fake_eeg_data.shape[0])
    
    # Create combined heatmap data
    # Shape: (n_samples * num_channels, time_samples_to_plot)
    combined_data = []
    y_labels = []
    
    for sample_idx in range(n_samples):
        for ch_idx in range(num_channels):
            combined_data.append(fake_eeg_data[sample_idx, :time_samples_to_plot, ch_idx])
            y_labels.append(f"S{sample_idx+1}-{channel_names[ch_idx]}")
    
    combined_data = np.array(combined_data)
    
    # Create time labels
    time_ticks = np.arange(0, time_samples_to_plot, int(sfreq/4))
    time_labels = [f"{t/sfreq:.2f}" for t in time_ticks]
    
    plt.figure(figsize=(15, max(8, n_samples * num_channels * 0.3)))
    im = plt.imshow(combined_data, 
                    aspect='auto', 
                    cmap='viridis', 
                    interpolation='bilinear')
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sample-Channel")
    plt.title(f"Multi-Sample Fake EEG Heatmap ({n_samples} samples, {n_seconds}s each)")
    plt.yticks(range(len(y_labels)), y_labels, fontsize=8)
    plt.xticks(time_ticks, time_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Amplitude')
    if save_plots:
        plot_path = os.path.join(output_dir, "multiple-sample-heatmaps.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    plt.tight_layout()
    plt.show()

def plot_channel_correlation_heatmap(real_eeg_data, fake_eeg_data, channel_names=None):
    """
    Plots correlation heatmaps between channels for real and fake EEG data.
    """
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(real_eeg_data.shape[0])]
    
    # Calculate correlation matrices
    real_corr = np.corrcoef(real_eeg_data)
    
    # For fake data, average across all samples first
    fake_avg = np.mean(fake_eeg_data, axis=0).T  # Shape: (num_channels, sequence_length)
    fake_corr = np.corrcoef(fake_avg)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Real EEG correlation heatmap
    sns.heatmap(real_corr, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                square=True,
                xticklabels=channel_names,
                yticklabels=channel_names,
                ax=ax1,
                fmt='.2f')
    ax1.set_title("(a)")
    
    # Fake EEG correlation heatmap
    sns.heatmap(fake_corr, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                square=True,
                xticklabels=channel_names,
                yticklabels=channel_names,
                ax=ax2,
                fmt='.2f')
    ax2.set_title("(b)")
    if save_plots:
        plot_path = os.path.join(output_dir, "channel-cor-relation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    plt.tight_layout()
    plt.show()

# Example usage:
if 'all_eeg_data_list' in globals() and all_eeg_data_list:
    sfreq = file_metadata[0]['sampling_frequency']  # Sampling frequency from real EEG metadata
    
    # 1. Side-by-side heatmap comparison
    plot_real_and_fake_eeg_heatmaps(
        real_eeg_data=scaled_segmented_eeg_data[0],  # Shape: (num_channels, total_samples)
        fake_eeg_data=fake_eeg_signals,              # Shape: (num_samples, sequence_length, num_channels)
        sfreq=sfreq,
        n_seconds=5,
        channel_names=eeg_channels_of_interest,
        dataset_index=0,
        sample_index=0
    )
    
    # 2. Multi-sample heatmap
    plot_multi_sample_heatmap(
        fake_eeg_data=fake_eeg_signals,
        sfreq=sfreq,
        n_samples=5,
        n_seconds=2,
        channel_names=eeg_channels_of_interest
    )
    
    # 3. Channel correlation comparison
    plot_channel_correlation_heatmap(
        real_eeg_data=scaled_segmented_eeg_data[0],
        fake_eeg_data=fake_eeg_signals,
        channel_names=eeg_channels_of_interest
    )
