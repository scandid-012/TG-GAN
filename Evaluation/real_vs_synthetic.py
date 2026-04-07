import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_real_and_fake_eeg(real_eeg_data, fake_eeg_data, sfreq, n_seconds=2, channel_names=None, dataset_index=0, sample_index=0):
    """
    Plots real and fake EEG segments side-by-side for comparison.
    save_plots=True
output_dir="/kaggle/working/"
    Parameters:
    - real_eeg_data: numpy array of shape (num_channels, total_samples) for real EEG
    - fake_eeg_data: numpy array of shape (num_samples, sequence_length, num_channels) for fake EEG
    - sfreq: Sampling frequency (Hz)
    - n_seconds: Duration to plot (seconds)
    - channel_names: List of channel names (optional)
    - dataset_index: Index of the real EEG dataset (for title)
    - sample_index: Index of the fake EEG sample to plot
    """
    save_plots=True
    output_dir="/kaggle/working/"
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(real_eeg_data.shape[0])]
    
    num_channels = real_eeg_data.shape[0]
    time_samples_to_plot = int(n_seconds * sfreq)
    
    # Ensure we don't exceed available samples
    real_samples = real_eeg_data.shape[1]
    fake_samples = fake_eeg_data.shape[1]
    time_samples_to_plot = min(time_samples_to_plot, real_samples, fake_samples)
    
    time_vector = np.arange(time_samples_to_plot) / sfreq
    
    # Create subplots: left for real EEG, right for fake EEG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # Plot real EEG
    for i in range(num_channels):
        offset = i * np.std(real_eeg_data[i, :time_samples_to_plot]) * 3
        ax1.plot(time_vector, real_eeg_data[i, :time_samples_to_plot] + offset, label=channel_names[i])
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (arbitrary offset)")
    ax1.set_title(f"Real EEG - Dataset {dataset_index} (First 2s)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot fake EEG (transpose to match real EEG shape: num_channels, sequence_length)
    fake_eeg_sample = fake_eeg_data[sample_index].transpose(1, 0)  # Shape: (num_channels, sequence_length)
    for i in range(num_channels):
        offset = i * np.std(fake_eeg_sample[i, :time_samples_to_plot]) * 3
        ax2.plot(time_vector, fake_eeg_sample[i, :time_samples_to_plot] + offset, label=channel_names[i])
    
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Fake EEG - Sample {sample_index} (First 2s)")
    ax2.legend()
    ax2.grid(True)
    if save_plots:
        plot_path = os.path.join(output_dir, "fakeVsReal-eegsample.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    plt.tight_layout()
    plt.show()

# Example usage:
if all_eeg_data_list:
    sfreq = file_metadata[0]['sampling_frequency']  # Sampling frequency from real EEG metadata
    # Plot real EEG from dataset 0 and fake EEG from sample 0
    plot_real_and_fake_eeg(
        real_eeg_data=scaled_segmented_eeg_data[0],  # Shape: (num_channels, total_samples)
        fake_eeg_data=fake_eeg_signals,      # Shape: (num_samples, sequence_length, num_channels)
        sfreq=sfreq,
        n_seconds=5,
        channel_names=eeg_channels_of_interest,
        dataset_index=0,
        sample_index=0
    )
