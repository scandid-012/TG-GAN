if raw:
    # Plot a segment of the raw data
    # Plot a few seconds, e.g., 10 seconds
    # You can specify which channels to plot using picks
    # Example: picks=['C3', 'Cz', 'C4'] if these are your EEG channels of interest
    print("\nPlotting raw data (first 10 seconds, first 5 channels)...")
    try:
        raw.plot(duration=10, n_channels=5, scalings='auto', block=False) # block=False for non-blocking plot in scripts
        plt.show() # In Kaggle, this might be needed, or plots might appear automatically
    except Exception as e:
        print(f"Could not plot raw data: {e}")

    # You can also plot specific channels:
    # eeg_channel_names = [ch for ch in raw.info['ch_names'] if 'EEG' in ch.upper() or ch in ['C3', 'Cz', 'C4']] # Heuristic
    # if not eeg_channel_names and len(raw.info['ch_names']) >=3: # Fallback if names are generic
    #    eeg_channel_names = raw.info['ch_names'][:3] # Assuming first 3 are EEG

    # if eeg_channel_names:
    #    print(f"\nPlotting specific EEG channels: {eeg_channel_names}")
    #    raw.plot(picks=eeg_channel_names, duration=10, scalings=dict(eeg=20e-6), block=False) # Typical EEG scaling
    #    plt.show()
    # else:
    #    print("Could not determine specific EEG channels for plotting by name.")

else:
    print("Raw data not loaded, skipping plotting.")
