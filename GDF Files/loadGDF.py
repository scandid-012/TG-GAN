 #Select one GDF file to explore
if gdf_files:
    # Let's pick the first GDF file for now
    example_gdf_file = gdf_files[0]
    print(f"Loading GDF file: {example_gdf_file}")

    # Load the GDF file
    # You might need to adjust preload=True based on memory constraints for very large files
    try:
        raw = mne.io.read_raw_gdf(example_gdf_file, preload=True, verbose='WARNING')
        print("Successfully loaded the GDF file.")
    except Exception as e:
        print(f"Error loading GDF file: {e}")
        print("You might need to install a specific GDF reader or check the file integrity.")
        # If standard MNE fails, sometimes specific GDF readers might be needed,
        # but for BCI competition data, MNE usually works.
else:
    print("No GDF files found. Please check the dataset path and file names.")
    raw = None # Define raw as None if no file is loaded
