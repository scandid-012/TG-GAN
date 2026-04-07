from scipy.signal import welch
import matplotlib.pyplot as plt

fs = SAMPLING_FREQUENCY  # 250 Hz
plt.figure(figsize=(12, 6))
for ch in range(num_eeg_channels):
    # Average PSD for real EEG
    freqs, psd_real = welch(scaled_segmented_eeg_data[:, :, ch].flatten(), fs=fs, nperseg=fs)
    plt.semilogy(freqs, psd_real, label=f'Real Channel {ch+1}')
    # Average PSD for generated EEG
    freqs, psd_fake = welch(fake_eeg_signals[:, :, ch].flatten(), fs=fs, nperseg=fs)
    plt.semilogy(freqs, psd_fake, label=f'Fake Channel {ch+1}', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Power Spectral Density: Real vs. Generated EEG')
plt.legend()
if save_plots:
        plot_path = os.path.join(output_dir, "PSD.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.grid(True)
plt.show()
