import matplotlib.pyplot as plt

def plot_eeg_segment(eeg_data, sfreq, dataset_index, n_seconds=5, channel_names=None):
    """Plots a segment of EEG data."""
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(eeg_data.shape[0])]
    
    num_channels, total_samples = eeg_data.shape
    time_samples_to_plot = int(n_seconds * sfreq)
    if total_samples < time_samples_to_plot:
        time_samples_to_plot = total_samples # Plot all if shorter than n_seconds
        
    time_vector = np.arange(time_samples_to_plot) / sfreq
    
    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        # Offset channels vertically for better visualization
        plt.plot(time_vector, eeg_data[i, :time_samples_to_plot] + (i * np.std(eeg_data[i, :time_samples_to_plot]) * 3), label=channel_names[i])
        
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arbitrary offset)")
    plt.title(f"EEG Data Segment - Dataset {dataset_index} (First {n_seconds}s)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot a segment from the first dataset
if all_eeg_data_list:
    sfreq = file_metadata[0]['sampling_frequency'] # Get sampling frequency from metadata
    # Using the eeg_channels_of_interest defined earlier
    plot_eeg_segment(all_eeg_data_list[0], sfreq, dataset_index=0, channel_names=eeg_channels_of_interest)

    # Optionally, plot from another dataset, e.g., the 5th one (index 4)
    if len(all_eeg_data_list) > 4:
        sfreq_other = file_metadata[4]['sampling_frequency']
        plot_eeg_segment(all_eeg_data_list[4], sfreq_other, dataset_index=4, channel_names=eeg_channels_of_interest)
