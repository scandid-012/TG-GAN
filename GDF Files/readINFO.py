if raw:
    # Print basic information
    print(raw)
    print("\nInformation about the data (info object):")
    print(raw.info)

    # Number of channels
    print(f"\nNumber of channels: {raw.info['nchan']}")

    # Channel names
    print(f"\nChannel names: {raw.info['ch_names']}")

    # Sampling frequency
    sfreq = raw.info['sfreq']
    print(f"\nSampling frequency: {sfreq} Hz")

    # Duration of the recording
    duration = raw.n_times / sfreq
    print(f"\nDuration of recording: {duration:.2f} seconds")

    # Get channel types (EEG, EOG, EMG, STIM, etc.)
    # MNE tries to infer them, but you might need to set them manually for some channels
    print(f"\nChannel types: {raw.get_channel_types(unique=True)}")

    # You might need to set channel types if they are not correctly inferred
    # For this dataset, the first few channels are EEG, and the last few are often EOG
    # Example (ADJUST BASED ON ACTUAL DATASET DOCUMENTATION):
    # eeg_channels = raw.info['ch_names'][:3] # Assuming first 3 are EEG
    # eog_channels = raw.info['ch_names'][3:6] # Assuming next 3 are EOG
    # mapping = {}
    # for ch in eeg_channels: mapping[ch] = 'eeg'
    # for ch in eog_channels: mapping[ch] = 'eog'
    # raw.set_channel_types(mapping)
    # print(f"Updated channel types: {raw.get_channel_types(unique=True)}")
else:
    print("Raw data not loaded, skipping information display.")
