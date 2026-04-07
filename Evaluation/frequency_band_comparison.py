import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Parameters
fs = 250  # Sampling frequency (Hz)
bands = {
    'Alpha (8–13 Hz)': (8, 13),
    'Beta (13–30 Hz)': (13, 30),
    'Gamma (30–100 Hz)': (30, 100)
}
num_channels = 3

# Select a sample EEG signal
real_sample = scaled_segmented_eeg_data[0]    # shape (3, 500)
fake_sample = fake_eeg_signals[0].T           # reshape (3, 500) to match real

# Band power computation
def compute_band_power(data, fs, band):
    f, psd = welch(data, fs=fs, nperseg=256)
    freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.sum(psd[:, idx_band], axis=1) * freq_res
    return band_power

# Prepare band powers
real_band_powers = {}
fake_band_powers = {}

for band_name, band_range in bands.items():
    real_band_powers[band_name] = compute_band_power(real_sample, fs, band_range)
    fake_band_powers[band_name] = compute_band_power(fake_sample, fs, band_range)

# Plot all in one row
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
channels = [f'Ch {i+1}' for i in range(num_channels)]
x = np.arange(num_channels)
width = 0.35

for idx, (band_name, _) in enumerate(bands.items()):
    ax = axes[idx]
    ax.bar(x - width/2, real_band_powers[band_name], width, label='Real')
    ax.bar(x + width/2, fake_band_powers[band_name], width, label='Fake')
    ax.set_title(band_name)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    if idx == 0:
        ax.set_ylabel('Power (uV²/Hz)')
    ax.legend()

plt.tight_layout()
if save_plots:
        plot_path = os.path.join(output_dir, "band-power-analysisn.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
