!pip install frechetdist
import numpy as np
import matplotlib.pyplot as plt
from frechetdist import frdist

# Parameters
num_fake = fake_eeg_signals.shape[0]
sequence_length = fake_eeg_signals.shape[1]
num_channels = fake_eeg_signals.shape[2]

# Match number of real samples
random_indices = np.random.choice(scaled_segmented_eeg_data.shape[0], num_fake, replace=False)
real_samples = scaled_segmented_eeg_data[random_indices]  # shape (5, 3, 500)

# Transpose to shape (5, 500, 3) to match fake
real_samples = np.transpose(real_samples, (0, 2, 1))

# Fréchet distance matrix: (samples, channels)
frechet_distances = np.zeros((num_fake, num_channels))

for i in range(num_fake):
    for ch in range(num_channels):
        real_curve = np.column_stack((np.arange(sequence_length), real_samples[i, :, ch]))
        fake_curve = np.column_stack((np.arange(sequence_length), fake_eeg_signals[i, :, ch]))
        frechet_distances[i, ch] = frdist(real_curve, fake_curve)

# Average across samples
mean_distances = np.mean(frechet_distances, axis=0)

# Plot
channels = np.arange(1, num_channels + 1)
plt.figure(figsize=(8, 5))
plt.bar(channels, mean_distances, color='orange', edgecolor='black')
plt.xticks(channels)
plt.xlabel("EEG Channel")
plt.ylabel("Mean Fréchet Distance")
plt.title("Fréchet Distance Between Real and Fake EEG (Per Channel)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
if save_plots:
        plot_path = os.path.join(output_dir, "FD.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
