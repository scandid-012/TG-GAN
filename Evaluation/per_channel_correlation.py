import numpy as np
import matplotlib.pyplot as plt

# Parameters
sequence_length = 500
max_lag = 100
fs = SAMPLING_FREQUENCY  # Set your sampling frequency, e.g., 250
num_channels = scaled_segmented_eeg_data.shape[1]

def compute_autocorr_per_channel(data, max_lag, is_fake=False):
    # Output: shape (channels, max_lag)
    channel_autocorr = []

    for ch in range(num_channels):
        acc = []
        for i in range(len(data)):
            if is_fake:
                signal = data[i, :, ch]  # shape: (time,)
            else:
                signal = data[i, ch, :]  # shape: (time,)
            corr = np.correlate(signal, signal, mode='full')
            mid = len(corr) // 2
            acc.append(corr[mid:mid+max_lag])
        acc = np.stack(acc)
        channel_autocorr.append(np.mean(acc, axis=0))
    return np.stack(channel_autocorr)

# Compute autocorrelations
autocorr_real = compute_autocorr_per_channel(scaled_segmented_eeg_data, max_lag)
autocorr_fake = compute_autocorr_per_channel(fake_eeg_signals, max_lag, is_fake=True)

lags = np.arange(0, max_lag) / fs

# Plotting
plt.figure(figsize=(15, 4 * num_channels))
for ch in range(num_channels):
    plt.subplot(num_channels, 1, ch + 1)
    plt.plot(lags, autocorr_real[ch], label=f'Real EEG - Channel {ch+1}')
    plt.plot(lags, autocorr_fake[ch], label=f'Fake EEG - Channel {ch+1}', linestyle='--')
    plt.xlabel('Lag (s)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Comparison - Channel {ch+1}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
if save_plots:
        plot_path = os.path.join(output_dir, "per channel-auto-correlation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
