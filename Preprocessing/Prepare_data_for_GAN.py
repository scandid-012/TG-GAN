
def create_segments(eeg_data_list, segment_length_samples, overlap_samples=0):
    all_segments = []
    for eeg_data in eeg_data_list: # eeg_data is (channels, timepoints)
        num_channels, total_samples = eeg_data.shape

        step_size = segment_length_samples - overlap_samples

        for i in range(0, total_samples - segment_length_samples + 1, step_size):
            segment = eeg_data[:, i : i + segment_length_samples]
            all_segments.append(segment)

    if not all_segments:
        return np.array([]) # Return empty array if no segments created
    return np.stack(all_segments) # Shape: (num_segments, num_channels, segment_length_samples)

# --- Parameters for segmentation ---
SAMPLING_FREQUENCY = 250.0 # Hz (from your metadata)
SEGMENT_DURATION_SECONDS = 2 # Example: 2-second segments
segment_length_samples = int(SEGMENT_DURATION_SECONDS * SAMPLING_FREQUENCY)

# Example: 50% overlap
overlap_duration_seconds = 1.0
overlap_samples = int(overlap_duration_seconds * SAMPLING_FREQUENCY)

# If you don't want overlap, set overlap_samples = 0
# overlap_samples = 0

print(f"Segment length: {segment_length_samples} samples ({SEGMENT_DURATION_SECONDS}s)")
print(f"Overlap: {overlap_samples} samples ({overlap_duration_seconds if overlap_samples > 0 else 0}s)")

# Create segments from your extracted data
# all_eeg_data_list should be your list of (3, timepoints) arrays
segmented_eeg_data = create_segments(all_eeg_data_list, segment_length_samples, overlap_samples)

if segmented_eeg_data.size > 0:
    print(f"Total number of segments created: {segmented_eeg_data.shape[0]}")
    print(f"Shape of segmented data: {segmented_eeg_data.shape}") # (num_segments, 3_channels, segment_length_samples)
else:
    print("No segments were created. Check segment_length_samples vs. data length or overlap settings.")

def scale_data(data_array, new_min=-1, new_max=1):
    """Scales data to [new_min, new_max] range."""
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    if max_val == min_val: # Avoid division by zero if data is flat
        return np.full(data_array.shape, new_min if min_val <=0 else (new_min+new_max)/2.0 ) # Or handle as error

    # Scale to [0,1]
    scaled_data = (data_array - min_val) / (max_val - min_val)
    # Then scale to [new_min, new_max]
    scaled_data = scaled_data * (new_max - new_min) + new_min
    return scaled_data, min_val, max_val # Return min/max for inverse scaling later

if segmented_eeg_data.size > 0:
    # Important: Calculate min/max over the entire segmented dataset for consistent scaling
    global_min = np.min(segmented_eeg_data)
    global_max = np.max(segmented_eeg_data)
    print(f"Global Min: {global_min}, Global Max: {global_max}")

    # Apply scaling
    # scaled_segmented_eeg_data = (segmented_eeg_data - global_min) / (global_max - global_min) # Scales to [0,1]
    # scaled_segmented_eeg_data = scaled_segmented_eeg_data * 2 - 1 # Scales to [-1,1]

    # Using the function for clarity and to store scaling params
    scaled_segmented_eeg_data, original_min, original_max = scale_data(segmented_eeg_data, new_min=-1, new_max=1)

    print(f"Shape of scaled data: {scaled_segmented_eeg_data.shape}")
    print(f"Min of scaled data: {np.min(scaled_segmented_eeg_data):.2f}")
    print(f"Max of scaled data: {np.max(scaled_segmented_eeg_data):.2f}")

    # Keep original_min and original_max if you need to revert the scaling later
    # (e.g., after generating fake samples, to see them in original EEG units)
else:
    print("Scaled data not computed as segmented_eeg_data is empty.")
    scaled_segmented_eeg_data = np.array([])
