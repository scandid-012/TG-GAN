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
