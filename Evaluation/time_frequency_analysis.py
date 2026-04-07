from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
save_plots=True
output_dir="/kaggle/working/"
fs = 250
# scaled_segmented_eeg_data: shape (94527, 3, 500)
# fake_eeg_signals: shape (5, 500, 3)

num_eeg_channels = scaled_segmented_eeg_data.shape[1]
sequence_length = scaled_segmented_eeg_data.shape[2]

# Pick the first sample from real and fake for plotting
real_sample = scaled_segmented_eeg_data[0]  # shape (3, 500)
fake_sample = fake_eeg_signals[0]           # shape (500, 3)

plt.figure(figsize=(12, 4 * num_eeg_channels))

for ch in range(num_eeg_channels):
    # Real EEG
    f_real, t_real, Sxx_real = spectrogram(real_sample[ch], fs=fs)
    plt.subplot(num_eeg_channels, 2, 2 * ch + 1)
    plt.pcolormesh(t_real, f_real, 10 * np.log10(Sxx_real + 1e-10), shading='gouraud')
    plt.title(f'Real EEG - Channel {ch+1}')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='dB')

    # Fake EEG
    f_fake, t_fake, Sxx_fake = spectrogram(fake_sample[:, ch], fs=fs)
    plt.subplot(num_eeg_channels, 2, 2 * ch + 2)
    plt.pcolormesh(t_fake, f_fake, 10 * np.log10(Sxx_fake + 1e-10), shading='gouraud')
    plt.title(f'Fake EEG - Channel {ch+1}')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='dB')

plt.tight_layout()
if save_plots:
        plot_path = os.path.join(output_dir, "time-freq-analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
