# Example shapes
# real_samples.shape = (100, 500, 3)
# fake_eeg_signals.shape = (100, 500, 3)

num_samples, time_steps, num_channels = real_samples.shape
frechet_distances = np.zeros((num_samples, num_channels))

# Compute Fréchet distance per sample per channel
for i in range(num_samples):
    for ch in range(num_channels):
        real_curve = np.column_stack((np.arange(time_steps), real_samples[i, :, ch]))
        fake_curve = np.column_stack((np.arange(time_steps), fake_eeg_signals[i, :, ch]))
        frechet_distances[i, ch] = frdist(real_curve, fake_curve)

# Plot
plt.figure(figsize=(12, 6))
for ch in range(num_channels):
    plt.plot(range(num_samples), frechet_distances[:, ch], label=f'Channel {ch+1}')

plt.title('Fréchet Distance per Sample for Each EEG Channel')
plt.xlabel('Sample Index')
plt.ylabel('Fréchet Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
if save_plots:
        plot_path = os.path.join(output_dir, "FD_persample.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
plt.show()
