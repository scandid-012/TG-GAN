import numpy as np

def compute_autocorr(data, max_lag=100, data_format='real'):
    """
    Computes average normalized autocorrelation for each channel.
    
    Parameters:
    - data: numpy array of EEG data
    - max_lag: number of lags to compute
    - data_format: 'real' for (samples, channels, time), 'fake' for (samples, time, channels)
    
    Returns:
    - mean autocorrelation array of shape (max_lag,)
    """
    autocorr = []

    for sample in data:
        if data_format == 'real':  # shape = (samples, channels, time)
            for ch in range(sample.shape[0]):
                signal = sample[ch, :]
        elif data_format == 'fake':  # shape = (samples, time, channels)
            for ch in range(sample.shape[1]):
                signal = sample[:, ch]
        else:
            raise ValueError("Invalid data_format. Use 'real' or 'fake'.")

            # Full autocorrelation
            full_corr = np.correlate(signal, signal, mode='full')
            center = len(full_corr) // 2
            segment = full_corr[center:center+max_lag]

            # Normalize
            segment = segment / full_corr[center] if full_corr[center] != 0 else np.zeros(max_lag)
            autocorr.append(segment)

    return np.mean(autocorr, axis=0)
    import numpy as np
import matplotlib.pyplot as plt

# Parameters
max_lag = 100
SAMPLING_FREQUENCY = 250  # update this if different

# Function
def compute_autocorr(data, max_lag=100, data_format='real'):
    autocorr = []
    if data_format == 'real':  # (samples, channels, time)
        for sample in data:
            for ch in range(sample.shape[0]):
                signal = sample[ch, :]
                if np.all(signal == 0): continue  # skip empty/zero signals
                full_corr = np.correlate(signal, signal, mode='full')
                center = len(full_corr) // 2
                segment = full_corr[center:center + max_lag]
                if full_corr[center] != 0:
                    segment = segment / full_corr[center]
                    autocorr.append(segment)
    elif data_format == 'fake':  # (samples, time, channels)
        for sample in data:
            for ch in range(sample.shape[1]):
                signal = sample[:, ch]
                if np.all(signal == 0): continue
                full_corr = np.correlate(signal, signal, mode='full')
                center = len(full_corr) // 2
                segment = full_corr[center:center + max_lag]
                if full_corr[center] != 0:
                    segment = segment / full_corr[center]
                    autocorr.append(segment)
    else:
        raise ValueError("data_format must be 'real' or 'fake'")
    
    if len(autocorr) == 0:
        return np.zeros(max_lag)
    
    return np.mean(autocorr, axis=0)

# Run computation
autocorr_real = compute_autocorr(scaled_segmented_eeg_data, max_lag=max_lag, data_format='real')
autocorr_fake = compute_autocorr(fake_eeg_signals, max_lag=max_lag, data_format='fake')

# Lags in seconds
lags = np.arange(0, max_lag) / SAMPLING_FREQUENCY

# Plot
plt.figure(figsize=(10, 5))
plt.plot(lags, autocorr_real, label='Real EEG Autocorrelation')
plt.plot(lags, autocorr_fake, label='Fake EEG Autocorrelation', linestyle='--')
plt.xlabel('Lag (s)')
plt.ylabel('Normalized Autocorrelation')
plt.title('Autocorrelation Comparison')
plt.legend()
plt.grid(True)
if save_plots:
        plot_path = os.path.join(output_dir, "auto-correlation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
